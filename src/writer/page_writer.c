/**
 * @file page_writer.c
 * @brief Data page and dictionary page creation
 *
 * Handles encoding values into pages with proper headers,
 * definition/repetition levels, and compression.
 */

#include <carquet/carquet.h>
#include <carquet/error.h>
#include "core/buffer.h"
#include "encoding/plain.h"
#include "encoding/rle.h"
#include "thrift/thrift_decode.h"
#include "thrift/thrift_encode.h"
#include "thrift/parquet_types.h"
#include <stdlib.h>
#include <string.h>

/* Forward declarations for compression */
extern carquet_status_t carquet_snappy_compress(const uint8_t* src, size_t src_size,
                                                 uint8_t* dst, size_t dst_capacity,
                                                 size_t* dst_size);
extern size_t carquet_snappy_compress_bound(size_t src_size);

/* CRC32 for page integrity verification */
extern uint32_t carquet_crc32(const uint8_t* data, size_t length);

extern carquet_status_t carquet_lz4_compress(const uint8_t* src, size_t src_size,
                                              uint8_t* dst, size_t dst_capacity,
                                              size_t* dst_size);
extern size_t carquet_lz4_compress_bound(size_t src_size);

extern int carquet_gzip_compress(const uint8_t* src, size_t src_size,
                                  uint8_t* dst, size_t dst_capacity,
                                  size_t* dst_size, int level);
extern size_t carquet_gzip_compress_bound(size_t src_size);

extern int carquet_zstd_compress(const uint8_t* src, size_t src_size,
                                  uint8_t* dst, size_t dst_capacity,
                                  size_t* dst_size, int level);
extern size_t carquet_zstd_compress_bound(size_t src_size);

/* ============================================================================
 * Page Writer Structure
 * ============================================================================
 */

typedef struct carquet_page_writer {
    carquet_buffer_t values_buffer;      /* Encoded values */
    carquet_buffer_t def_levels_buffer;  /* Definition levels (RLE) */
    carquet_buffer_t rep_levels_buffer;  /* Repetition levels (RLE) */
    carquet_buffer_t page_buffer;        /* Final page with header */
    carquet_buffer_t compress_buffer;   /* Reusable compression buffer */

    carquet_physical_type_t type;
    carquet_encoding_t encoding;
    carquet_compression_t compression;

    int16_t max_def_level;
    int16_t max_rep_level;
    int32_t type_length;  /* For FIXED_LEN_BYTE_ARRAY */

    int64_t num_values;
    int64_t num_nulls;

    int32_t compression_level;   /* 0 = use codec default */

    /* Options */
    bool write_crc;          /* Compute and write CRC32 for pages */
    bool write_statistics;   /* Write min/max statistics in page header */

    /* Statistics tracking */
    bool has_min_max;
    uint8_t min_value[64];
    uint8_t max_value[64];
    size_t min_max_size;
} carquet_page_writer_t;

/* Forward declaration for internal use */
void carquet_page_writer_destroy(carquet_page_writer_t* writer);

/* ============================================================================
 * Page Writer Lifecycle
 * ============================================================================
 */

carquet_page_writer_t* carquet_page_writer_create(
    carquet_physical_type_t type,
    carquet_encoding_t encoding,
    carquet_compression_t compression,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length,
    int32_t compression_level) {

    carquet_page_writer_t* writer = calloc(1, sizeof(carquet_page_writer_t));
    if (!writer) return NULL;

    carquet_buffer_init(&writer->values_buffer);
    carquet_buffer_init(&writer->def_levels_buffer);
    carquet_buffer_init(&writer->rep_levels_buffer);
    carquet_buffer_init(&writer->page_buffer);
    carquet_buffer_init(&writer->compress_buffer);

    writer->type = type;
    writer->encoding = encoding;
    writer->compression = compression;
    writer->max_def_level = max_def_level;
    writer->max_rep_level = max_rep_level;
    writer->type_length = type_length;
    writer->compression_level = compression_level;
    writer->write_crc = true;         /* Enable CRC by default for integrity */
    writer->write_statistics = true;  /* Enable statistics by default for pushdown */

    return writer;
}

void carquet_page_writer_destroy(carquet_page_writer_t* writer) {
    if (writer) {
        carquet_buffer_destroy(&writer->values_buffer);
        carquet_buffer_destroy(&writer->def_levels_buffer);
        carquet_buffer_destroy(&writer->rep_levels_buffer);
        carquet_buffer_destroy(&writer->page_buffer);
        carquet_buffer_destroy(&writer->compress_buffer);
        free(writer);
    }
}

void carquet_page_writer_reset(carquet_page_writer_t* writer) {
    carquet_buffer_clear(&writer->values_buffer);
    carquet_buffer_clear(&writer->def_levels_buffer);
    carquet_buffer_clear(&writer->rep_levels_buffer);
    carquet_buffer_clear(&writer->page_buffer);
    carquet_buffer_clear(&writer->compress_buffer);
    writer->num_values = 0;
    writer->num_nulls = 0;
    writer->has_min_max = false;
}

/* ============================================================================
 * Level Encoding (RLE/Bit-Packed Hybrid)
 * ============================================================================
 */

static int bit_width_for_max(int16_t max_level) {
    if (max_level == 0) return 0;
    int width = 0;
    int16_t val = max_level;
    while (val > 0) {
        width++;
        val >>= 1;
    }
    return width;
}

static carquet_status_t encode_levels(
    const int16_t* levels,
    int64_t count,
    int16_t max_level,
    carquet_buffer_t* output) {

    if (max_level == 0) {
        return CARQUET_OK;
    }

    int bit_width = bit_width_for_max(max_level);
    size_t prefix_offset = output->size;
    carquet_status_t status = carquet_buffer_append_u32_le(output, 0);
    if (status != CARQUET_OK) {
        return status;
    }

    size_t encoded_offset = output->size;
    if (levels) {
        status = carquet_rle_encode_levels(levels, count, bit_width, output);
    } else {
        carquet_rle_encoder_t enc;
        carquet_rle_encoder_init(&enc, output, bit_width);
        status = carquet_rle_encoder_put_repeat(&enc, (uint32_t)max_level, count);
        if (status == CARQUET_OK) {
            status = carquet_rle_encoder_flush(&enc);
        }
    }

    if (status != CARQUET_OK) {
        output->size = prefix_offset;
        return status;
    }

    size_t encoded_size = output->size - encoded_offset;
    if (encoded_size > UINT32_MAX) {
        output->size = prefix_offset;
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    output->data[prefix_offset] = (uint8_t)(encoded_size & 0xFF);
    output->data[prefix_offset + 1] = (uint8_t)((encoded_size >> 8) & 0xFF);
    output->data[prefix_offset + 2] = (uint8_t)((encoded_size >> 16) & 0xFF);
    output->data[prefix_offset + 3] = (uint8_t)((encoded_size >> 24) & 0xFF);
    return CARQUET_OK;
}

/* ============================================================================
 * Statistics Tracking
 * ============================================================================
 */

static void update_statistics_i32(carquet_page_writer_t* writer,
                                   const int32_t* values, int64_t count) {
    if (count <= 0) return;
    int32_t min_v, max_v;
    int64_t start = 0;
    if (!writer->has_min_max) {
        min_v = values[0];
        max_v = values[0];
        start = 1;
        writer->has_min_max = true;
        writer->min_max_size = sizeof(int32_t);
    } else {
        memcpy(&min_v, writer->min_value, sizeof(min_v));
        memcpy(&max_v, writer->max_value, sizeof(max_v));
    }
    for (int64_t i = start; i < count; i++) {
        if (values[i] < min_v) min_v = values[i];
        if (values[i] > max_v) max_v = values[i];
    }
    memcpy(writer->min_value, &min_v, sizeof(min_v));
    memcpy(writer->max_value, &max_v, sizeof(max_v));
}

static void update_statistics_i64(carquet_page_writer_t* writer,
                                   const int64_t* values, int64_t count) {
    if (count <= 0) return;
    int64_t min_v, max_v;
    int64_t start = 0;
    if (!writer->has_min_max) {
        min_v = values[0];
        max_v = values[0];
        start = 1;
        writer->has_min_max = true;
        writer->min_max_size = sizeof(int64_t);
    } else {
        memcpy(&min_v, writer->min_value, sizeof(min_v));
        memcpy(&max_v, writer->max_value, sizeof(max_v));
    }
    for (int64_t i = start; i < count; i++) {
        if (values[i] < min_v) min_v = values[i];
        if (values[i] > max_v) max_v = values[i];
    }
    memcpy(writer->min_value, &min_v, sizeof(min_v));
    memcpy(writer->max_value, &max_v, sizeof(max_v));
}

static void update_statistics_float(carquet_page_writer_t* writer,
                                     const float* values, int64_t count) {
    if (count <= 0) return;
    float min_v, max_v;
    int64_t start = 0;
    if (!writer->has_min_max) {
        min_v = values[0];
        max_v = values[0];
        start = 1;
        writer->has_min_max = true;
        writer->min_max_size = sizeof(float);
    } else {
        memcpy(&min_v, writer->min_value, sizeof(min_v));
        memcpy(&max_v, writer->max_value, sizeof(max_v));
    }
    for (int64_t i = start; i < count; i++) {
        if (values[i] < min_v) min_v = values[i];
        if (values[i] > max_v) max_v = values[i];
    }
    memcpy(writer->min_value, &min_v, sizeof(min_v));
    memcpy(writer->max_value, &max_v, sizeof(max_v));
}

static void update_statistics_double(carquet_page_writer_t* writer,
                                      const double* values, int64_t count) {
    if (count <= 0) return;
    double min_v, max_v;
    int64_t start = 0;
    if (!writer->has_min_max) {
        min_v = values[0];
        max_v = values[0];
        start = 1;
        writer->has_min_max = true;
        writer->min_max_size = sizeof(double);
    } else {
        memcpy(&min_v, writer->min_value, sizeof(min_v));
        memcpy(&max_v, writer->max_value, sizeof(max_v));
    }
    for (int64_t i = start; i < count; i++) {
        if (values[i] < min_v) min_v = values[i];
        if (values[i] > max_v) max_v = values[i];
    }
    memcpy(writer->min_value, &min_v, sizeof(min_v));
    memcpy(writer->max_value, &max_v, sizeof(max_v));
}

/* ============================================================================
 * Value Encoding
 * ============================================================================
 */

carquet_status_t carquet_page_writer_add_values(
    carquet_page_writer_t* writer,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels) {

    if (!writer || !values) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    carquet_status_t status = CARQUET_OK;

    /* Count nulls and non-null values */
    int64_t num_non_null = num_values;
    if (def_levels && writer->max_def_level > 0) {
        num_non_null = 0;
        for (int64_t i = 0; i < num_values; i++) {
            if (def_levels[i] == writer->max_def_level) {
                num_non_null++;
            }
        }
        writer->num_nulls += (num_values - num_non_null);
    }

    /* Encode definition levels.
     * If def_levels is NULL for an OPTIONAL column, generate all-present levels
     * since Parquet requires definition levels for non-REQUIRED columns. */
    if (writer->max_def_level > 0) {
        status = encode_levels(def_levels, num_values, writer->max_def_level,
                               &writer->def_levels_buffer);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    /* Encode repetition levels */
    if (writer->max_rep_level > 0 && rep_levels) {
        encode_levels(rep_levels, num_values, writer->max_rep_level,
                      &writer->rep_levels_buffer);
    }

    /* Encode values using PLAIN encoding.
     *
     * The values array uses sparse encoding: it contains only non-null values
     * (packed at the front), with num_non_null entries. The def_levels array
     * has num_values entries (one per logical row) indicating which rows are
     * null vs present.
     */
    switch (writer->type) {
        case CARQUET_PHYSICAL_BOOLEAN: {
            const uint8_t* bools = (const uint8_t*)values;
            status = carquet_encode_plain_boolean(bools, num_non_null,
                                                   &writer->values_buffer);
            break;
        }

        case CARQUET_PHYSICAL_INT32: {
            const int32_t* ints = (const int32_t*)values;
            status = carquet_encode_plain_int32(ints, num_non_null,
                                                 &writer->values_buffer);
            update_statistics_i32(writer, ints, num_non_null);
            break;
        }

        case CARQUET_PHYSICAL_INT64: {
            const int64_t* ints = (const int64_t*)values;
            status = carquet_encode_plain_int64(ints, num_non_null,
                                                 &writer->values_buffer);
            update_statistics_i64(writer, ints, num_non_null);
            break;
        }

        case CARQUET_PHYSICAL_FLOAT: {
            const float* floats = (const float*)values;
            status = carquet_encode_plain_float(floats, num_non_null,
                                                 &writer->values_buffer);
            update_statistics_float(writer, floats, num_non_null);
            break;
        }

        case CARQUET_PHYSICAL_DOUBLE: {
            const double* doubles = (const double*)values;
            status = carquet_encode_plain_double(doubles, num_non_null,
                                                  &writer->values_buffer);
            update_statistics_double(writer, doubles, num_non_null);
            break;
        }

        case CARQUET_PHYSICAL_BYTE_ARRAY: {
            const carquet_byte_array_t* arrays = (const carquet_byte_array_t*)values;
            status = carquet_encode_plain_byte_array(arrays, num_non_null,
                                                      &writer->values_buffer);
            break;
        }

        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY: {
            const uint8_t* fixed = (const uint8_t*)values;
            status = carquet_encode_plain_fixed_byte_array(fixed, num_non_null,
                                                            writer->type_length,
                                                            &writer->values_buffer);
            break;
        }

        default:
            status = CARQUET_ERROR_NOT_IMPLEMENTED;
    }

    writer->num_values += num_values;
    return status;
}

/* ============================================================================
 * Compression
 * ============================================================================
 */

static carquet_status_t compress_data(
    carquet_compression_t codec,
    const uint8_t* input,
    size_t input_size,
    carquet_buffer_t* temp_buffer,
    const uint8_t** compressed_data,
    size_t* compressed_size,
    int32_t compression_level) {

    if (!compressed_data || !compressed_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    if (codec == CARQUET_COMPRESSION_UNCOMPRESSED) {
        *compressed_data = input;
        *compressed_size = input_size;
        return CARQUET_OK;
    }

    size_t bound = 0;
    switch (codec) {
        case CARQUET_COMPRESSION_SNAPPY:
            bound = carquet_snappy_compress_bound(input_size);
            break;
        case CARQUET_COMPRESSION_LZ4:
        case CARQUET_COMPRESSION_LZ4_RAW:
            bound = carquet_lz4_compress_bound(input_size);
            break;
        case CARQUET_COMPRESSION_GZIP:
            bound = carquet_gzip_compress_bound(input_size);
            break;
        case CARQUET_COMPRESSION_ZSTD:
            bound = carquet_zstd_compress_bound(input_size);
            break;
        default:
            return CARQUET_ERROR_UNSUPPORTED_CODEC;
    }

    /* Ensure temp buffer is large enough */
    if (temp_buffer->capacity < bound) {
        carquet_buffer_destroy(temp_buffer);
        carquet_status_t reserve_status = carquet_buffer_init_capacity(temp_buffer, bound);
        if (reserve_status != CARQUET_OK) {
            return reserve_status;
        }
    }
    uint8_t* compressed = temp_buffer->data;

    size_t local_compressed_size = 0;
    carquet_status_t status;

    switch (codec) {
        case CARQUET_COMPRESSION_SNAPPY:
            status = carquet_snappy_compress(input, input_size,
                                              compressed, bound, &local_compressed_size);
            break;
        case CARQUET_COMPRESSION_LZ4:
        case CARQUET_COMPRESSION_LZ4_RAW:
            status = carquet_lz4_compress(input, input_size,
                                           compressed, bound, &local_compressed_size);
            break;
        case CARQUET_COMPRESSION_GZIP:
            status = carquet_gzip_compress(input, input_size,
                                            compressed, bound, &local_compressed_size,
                                            compression_level > 0 ? compression_level : 6);
            break;
        case CARQUET_COMPRESSION_ZSTD:
            status = carquet_zstd_compress(input, input_size,
                                            compressed, bound, &local_compressed_size,
                                            compression_level > 0 ? compression_level : 3);
            break;
        default:
            status = CARQUET_ERROR_UNSUPPORTED_CODEC;
    }

    if (status != CARQUET_OK) {
        return status;
    }

    temp_buffer->size = local_compressed_size;
    *compressed_data = compressed;
    *compressed_size = local_compressed_size;
    return CARQUET_OK;
}

/* ============================================================================
 * Page Finalization
 * ============================================================================
 */

carquet_status_t carquet_page_writer_finalize(
    carquet_page_writer_t* writer,
    const uint8_t** page_data,
    size_t* page_size,
    int32_t* uncompressed_size,
    int32_t* compressed_size) {

    if (!writer || !page_data || !page_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    carquet_buffer_clear(&writer->page_buffer);

    /* Build uncompressed page data: rep_levels + def_levels + values.
     * For REQUIRED columns (no levels), skip the intermediate buffer
     * and point directly at the values buffer to avoid a copy. */
    const uint8_t* unc_data;
    size_t unc_size;
    carquet_buffer_t uncompressed;
    bool has_levels = (writer->rep_levels_buffer.size > 0 ||
                       writer->def_levels_buffer.size > 0);

    if (has_levels) {
        carquet_buffer_init(&uncompressed);
        if (writer->rep_levels_buffer.size > 0) {
            carquet_buffer_append(&uncompressed,
                                   writer->rep_levels_buffer.data,
                                   writer->rep_levels_buffer.size);
        }
        if (writer->def_levels_buffer.size > 0) {
            carquet_buffer_append(&uncompressed,
                                   writer->def_levels_buffer.data,
                                   writer->def_levels_buffer.size);
        }
        carquet_buffer_append(&uncompressed,
                               writer->values_buffer.data,
                               writer->values_buffer.size);
        unc_data = uncompressed.data;
        unc_size = uncompressed.size;
    } else {
        unc_data = writer->values_buffer.data;
        unc_size = writer->values_buffer.size;
    }

    *uncompressed_size = (int32_t)unc_size;

    /* Compress if needed */
    const uint8_t* compressed_data = NULL;
    size_t compressed_data_size = 0;
    carquet_status_t status = compress_data(writer->compression,
                                             unc_data, unc_size,
                                             &writer->compress_buffer,
                                             &compressed_data,
                                             &compressed_data_size,
                                             writer->compression_level);

    if (status != CARQUET_OK) {
        if (has_levels) {
            carquet_buffer_destroy(&uncompressed);
        }
        return status;
    }

    *compressed_size = (int32_t)compressed_data_size;

    /* Compute CRC32 if enabled */
    uint32_t page_crc = 0;
    if (writer->write_crc) {
        page_crc = carquet_crc32(compressed_data, compressed_data_size);
    }

    /* Build page header using Thrift */
    thrift_encoder_t enc;
    thrift_encoder_init(&enc, &writer->page_buffer);

    /* PageHeader struct */
    thrift_write_struct_begin(&enc);

    /* Field 1: type (DATA_PAGE = 0) */
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
    thrift_write_i32(&enc, CARQUET_PAGE_DATA);

    /* Field 2: uncompressed_page_size */
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
    thrift_write_i32(&enc, *uncompressed_size);

    /* Field 3: compressed_page_size */
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 3);
    thrift_write_i32(&enc, *compressed_size);

    /* Field 4: crc (optional - write if enabled) */
    if (writer->write_crc) {
        thrift_write_field_header(&enc, THRIFT_TYPE_I32, 4);
        thrift_write_i32(&enc, (int32_t)page_crc);
    }

    /* Field 5: data_page_header (DataPageHeader struct) */
    thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 5);
    thrift_write_struct_begin(&enc);

    /* DataPageHeader field 1: num_values */
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
    thrift_write_i32(&enc, (int32_t)writer->num_values);

    /* DataPageHeader field 2: encoding */
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
    thrift_write_i32(&enc, (int32_t)writer->encoding);

    /* DataPageHeader field 3: definition_level_encoding (RLE) */
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 3);
    thrift_write_i32(&enc, CARQUET_ENCODING_RLE);

    /* DataPageHeader field 4: repetition_level_encoding (RLE) */
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 4);
    thrift_write_i32(&enc, CARQUET_ENCODING_RLE);

    /* DataPageHeader field 5: statistics (optional - write if enabled and available) */
    if (writer->write_statistics && writer->has_min_max) {
        thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 5);
        thrift_write_struct_begin(&enc);

        /* Statistics field 3: null_count */
        thrift_write_field_header(&enc, THRIFT_TYPE_I64, 3);
        thrift_write_i64(&enc, writer->num_nulls);

        /* Statistics field 5: max_value (binary) */
        thrift_write_field_header(&enc, THRIFT_TYPE_BINARY, 5);
        thrift_write_binary(&enc, writer->max_value, (int32_t)writer->min_max_size);

        /* Statistics field 6: min_value (binary) */
        thrift_write_field_header(&enc, THRIFT_TYPE_BINARY, 6);
        thrift_write_binary(&enc, writer->min_value, (int32_t)writer->min_max_size);

        thrift_write_struct_end(&enc);  /* End Statistics */
    }

    thrift_write_struct_end(&enc);  /* End DataPageHeader */
    thrift_write_struct_end(&enc);  /* End PageHeader */

    /* Append compressed data after header */
    status = carquet_buffer_append(&writer->page_buffer, compressed_data, compressed_data_size);
    if (has_levels) {
        carquet_buffer_destroy(&uncompressed);
    }
    if (status != CARQUET_OK) {
        return status;
    }

    *page_data = writer->page_buffer.data;
    *page_size = writer->page_buffer.size;

    return CARQUET_OK;
}

size_t carquet_page_writer_estimated_size(const carquet_page_writer_t* writer) {
    if (!writer) return 0;
    return writer->values_buffer.size +
           writer->def_levels_buffer.size +
           writer->rep_levels_buffer.size + 64;  /* Header overhead */
}

int64_t carquet_page_writer_num_values(const carquet_page_writer_t* writer) {
    return writer ? writer->num_values : 0;
}

/* ============================================================================
 * Options Configuration
 * ============================================================================
 */

void carquet_page_writer_set_crc(carquet_page_writer_t* writer, bool enabled) {
    if (writer) {
        writer->write_crc = enabled;
    }
}

void carquet_page_writer_set_statistics(carquet_page_writer_t* writer, bool enabled) {
    if (writer) {
        writer->write_statistics = enabled;
    }
}

/* ============================================================================
 * Statistics Retrieval (for column-level aggregation)
 * ============================================================================
 */

bool carquet_page_writer_get_statistics(
    const carquet_page_writer_t* writer,
    const uint8_t** min_value,
    const uint8_t** max_value,
    size_t* value_size,
    int64_t* null_count) {

    if (!writer || !writer->has_min_max) {
        return false;
    }

    if (min_value) *min_value = writer->min_value;
    if (max_value) *max_value = writer->max_value;
    if (value_size) *value_size = writer->min_max_size;
    if (null_count) *null_count = writer->num_nulls;

    return true;
}

int64_t carquet_page_writer_null_count(const carquet_page_writer_t* writer) {
    return writer ? writer->num_nulls : 0;
}
