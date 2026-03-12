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
extern uint32_t carquet_crc32_update(uint32_t crc, const uint8_t* data, size_t length);

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

extern carquet_status_t carquet_byte_stream_split_encode_float(
    const float* values,
    int64_t count,
    uint8_t* output,
    size_t output_capacity,
    size_t* bytes_written);
extern carquet_status_t carquet_byte_stream_split_encode_double(
    const double* values,
    int64_t count,
    uint8_t* output,
    size_t output_capacity,
    size_t* bytes_written);
extern int64_t carquet_dispatch_count_non_nulls(const int16_t* def_levels, int64_t count,
                                                 int16_t max_def_level);
extern void carquet_dispatch_minmax_i32(const int32_t* values, int64_t count,
                                         int32_t* min_value, int32_t* max_value);
extern void carquet_dispatch_minmax_i64(const int64_t* values, int64_t count,
                                         int64_t* min_value, int64_t* max_value);
extern void carquet_dispatch_minmax_float(const float* values, int64_t count,
                                           float* min_value, float* max_value);
extern void carquet_dispatch_minmax_double(const double* values, int64_t count,
                                            double* min_value, double* max_value);
extern void carquet_dispatch_copy_minmax_i32(const int32_t* values, int64_t count, int32_t* output,
                                              int32_t* min_value, int32_t* max_value);
extern void carquet_dispatch_copy_minmax_i64(const int64_t* values, int64_t count, int64_t* output,
                                              int64_t* min_value, int64_t* max_value);
extern void carquet_dispatch_copy_minmax_float(const float* values, int64_t count, float* output,
                                                float* min_value, float* max_value);
extern void carquet_dispatch_copy_minmax_double(const double* values, int64_t count, double* output,
                                                 double* min_value, double* max_value);

/* ============================================================================
 * Page Writer Structure
 * ============================================================================
 */

typedef struct carquet_page_writer {
    carquet_buffer_t values_buffer;      /* Encoded values */
    carquet_buffer_t def_levels_buffer;  /* Definition levels (RLE) */
    carquet_buffer_t rep_levels_buffer;  /* Repetition levels (RLE) */
    carquet_buffer_t staging_buffer;     /* Reusable page payload staging */
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
carquet_status_t carquet_page_writer_finalize_to_buffer(
    carquet_page_writer_t* writer,
    carquet_buffer_t* output_buffer,
    size_t* page_size,
    int32_t* uncompressed_size,
    int32_t* compressed_size);

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
    carquet_buffer_init(&writer->staging_buffer);
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
        carquet_buffer_destroy(&writer->staging_buffer);
        carquet_buffer_destroy(&writer->page_buffer);
        carquet_buffer_destroy(&writer->compress_buffer);
        free(writer);
    }
}

void carquet_page_writer_reset(carquet_page_writer_t* writer) {
    carquet_buffer_clear(&writer->values_buffer);
    carquet_buffer_clear(&writer->def_levels_buffer);
    carquet_buffer_clear(&writer->rep_levels_buffer);
    carquet_buffer_clear(&writer->staging_buffer);
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
    if (!writer->has_min_max) {
        int32_t min_v, max_v;
        carquet_dispatch_minmax_i32(values, count, &min_v, &max_v);
        writer->has_min_max = true;
        writer->min_max_size = sizeof(int32_t);
        memcpy(writer->min_value, &min_v, sizeof(min_v));
        memcpy(writer->max_value, &max_v, sizeof(max_v));
        return;
    }
    int32_t min_v, max_v;
    int32_t chunk_min, chunk_max;
    memcpy(&min_v, writer->min_value, sizeof(min_v));
    memcpy(&max_v, writer->max_value, sizeof(max_v));
    carquet_dispatch_minmax_i32(values, count, &chunk_min, &chunk_max);
    if (chunk_min < min_v) min_v = chunk_min;
    if (chunk_max > max_v) max_v = chunk_max;
    memcpy(writer->min_value, &min_v, sizeof(min_v));
    memcpy(writer->max_value, &max_v, sizeof(max_v));
}

static void update_statistics_i64(carquet_page_writer_t* writer,
                                   const int64_t* values, int64_t count) {
    if (count <= 0) return;
    if (!writer->has_min_max) {
        int64_t min_v, max_v;
        carquet_dispatch_minmax_i64(values, count, &min_v, &max_v);
        writer->has_min_max = true;
        writer->min_max_size = sizeof(int64_t);
        memcpy(writer->min_value, &min_v, sizeof(min_v));
        memcpy(writer->max_value, &max_v, sizeof(max_v));
        return;
    }
    int64_t min_v, max_v;
    int64_t chunk_min, chunk_max;
    memcpy(&min_v, writer->min_value, sizeof(min_v));
    memcpy(&max_v, writer->max_value, sizeof(max_v));
    carquet_dispatch_minmax_i64(values, count, &chunk_min, &chunk_max);
    if (chunk_min < min_v) min_v = chunk_min;
    if (chunk_max > max_v) max_v = chunk_max;
    memcpy(writer->min_value, &min_v, sizeof(min_v));
    memcpy(writer->max_value, &max_v, sizeof(max_v));
}

static void update_statistics_float(carquet_page_writer_t* writer,
                                     const float* values, int64_t count) {
    if (count <= 0) return;
    if (!writer->has_min_max) {
        float min_v, max_v;
        carquet_dispatch_minmax_float(values, count, &min_v, &max_v);
        writer->has_min_max = true;
        writer->min_max_size = sizeof(float);
        memcpy(writer->min_value, &min_v, sizeof(min_v));
        memcpy(writer->max_value, &max_v, sizeof(max_v));
        return;
    }
    float min_v, max_v;
    float chunk_min, chunk_max;
    memcpy(&min_v, writer->min_value, sizeof(min_v));
    memcpy(&max_v, writer->max_value, sizeof(max_v));
    carquet_dispatch_minmax_float(values, count, &chunk_min, &chunk_max);
    if (chunk_min < min_v) min_v = chunk_min;
    if (chunk_max > max_v) max_v = chunk_max;
    memcpy(writer->min_value, &min_v, sizeof(min_v));
    memcpy(writer->max_value, &max_v, sizeof(max_v));
}

static void update_statistics_double(carquet_page_writer_t* writer,
                                      const double* values, int64_t count) {
    if (count <= 0) return;
    if (!writer->has_min_max) {
        double min_v, max_v;
        carquet_dispatch_minmax_double(values, count, &min_v, &max_v);
        writer->has_min_max = true;
        writer->min_max_size = sizeof(double);
        memcpy(writer->min_value, &min_v, sizeof(min_v));
        memcpy(writer->max_value, &max_v, sizeof(max_v));
        return;
    }
    double min_v, max_v;
    double chunk_min, chunk_max;
    memcpy(&min_v, writer->min_value, sizeof(min_v));
    memcpy(&max_v, writer->max_value, sizeof(max_v));
    carquet_dispatch_minmax_double(values, count, &chunk_min, &chunk_max);
    if (chunk_min < min_v) min_v = chunk_min;
    if (chunk_max > max_v) max_v = chunk_max;
    memcpy(writer->min_value, &min_v, sizeof(min_v));
    memcpy(writer->max_value, &max_v, sizeof(max_v));
}

static carquet_status_t encode_plain_i32_with_stats(
    carquet_page_writer_t* writer,
    const int32_t* values,
    int64_t count) {
#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    size_t bytes_needed = (size_t)count * sizeof(int32_t);
    uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, bytes_needed);
    int32_t min_v, max_v;
    if (!dest) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }
    carquet_dispatch_copy_minmax_i32(values, count, (int32_t*)dest, &min_v, &max_v);
    if (!writer->has_min_max) {
        writer->has_min_max = true;
        writer->min_max_size = sizeof(min_v);
    } else {
        int32_t cur_min, cur_max;
        memcpy(&cur_min, writer->min_value, sizeof(cur_min));
        memcpy(&cur_max, writer->max_value, sizeof(cur_max));
        if (cur_min < min_v) min_v = cur_min;
        if (cur_max > max_v) max_v = cur_max;
    }
    memcpy(writer->min_value, &min_v, sizeof(min_v));
    memcpy(writer->max_value, &max_v, sizeof(max_v));
    return CARQUET_OK;
#else
    carquet_status_t status = carquet_encode_plain_int32(values, count, &writer->values_buffer);
    if (status == CARQUET_OK) {
        update_statistics_i32(writer, values, count);
    }
    return status;
#endif
}

static carquet_status_t encode_plain_i64_with_stats(
    carquet_page_writer_t* writer,
    const int64_t* values,
    int64_t count) {
#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    size_t bytes_needed = (size_t)count * sizeof(int64_t);
    uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, bytes_needed);
    int64_t min_v, max_v;
    if (!dest) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }
    carquet_dispatch_copy_minmax_i64(values, count, (int64_t*)dest, &min_v, &max_v);
    if (!writer->has_min_max) {
        writer->has_min_max = true;
        writer->min_max_size = sizeof(min_v);
    } else {
        int64_t cur_min, cur_max;
        memcpy(&cur_min, writer->min_value, sizeof(cur_min));
        memcpy(&cur_max, writer->max_value, sizeof(cur_max));
        if (cur_min < min_v) min_v = cur_min;
        if (cur_max > max_v) max_v = cur_max;
    }
    memcpy(writer->min_value, &min_v, sizeof(min_v));
    memcpy(writer->max_value, &max_v, sizeof(max_v));
    return CARQUET_OK;
#else
    carquet_status_t status = carquet_encode_plain_int64(values, count, &writer->values_buffer);
    if (status == CARQUET_OK) {
        update_statistics_i64(writer, values, count);
    }
    return status;
#endif
}

static carquet_status_t encode_plain_float_with_stats(
    carquet_page_writer_t* writer,
    const float* values,
    int64_t count) {
#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    size_t bytes_needed = (size_t)count * sizeof(float);
    uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, bytes_needed);
    float min_v, max_v;
    if (!dest) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }
    carquet_dispatch_copy_minmax_float(values, count, (float*)dest, &min_v, &max_v);
    if (!writer->has_min_max) {
        writer->has_min_max = true;
        writer->min_max_size = sizeof(min_v);
    } else {
        float cur_min, cur_max;
        memcpy(&cur_min, writer->min_value, sizeof(cur_min));
        memcpy(&cur_max, writer->max_value, sizeof(cur_max));
        if (cur_min < min_v) min_v = cur_min;
        if (cur_max > max_v) max_v = cur_max;
    }
    memcpy(writer->min_value, &min_v, sizeof(min_v));
    memcpy(writer->max_value, &max_v, sizeof(max_v));
    return CARQUET_OK;
#else
    carquet_status_t status = carquet_encode_plain_float(values, count, &writer->values_buffer);
    if (status == CARQUET_OK) {
        update_statistics_float(writer, values, count);
    }
    return status;
#endif
}

static carquet_status_t encode_plain_double_with_stats(
    carquet_page_writer_t* writer,
    const double* values,
    int64_t count) {
#if CARQUET_LITTLE_ENDIAN && !defined(CARQUET_STRICT_ALIGN)
    size_t bytes_needed = (size_t)count * sizeof(double);
    uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, bytes_needed);
    double min_v, max_v;
    if (!dest) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }
    carquet_dispatch_copy_minmax_double(values, count, (double*)dest, &min_v, &max_v);
    if (!writer->has_min_max) {
        writer->has_min_max = true;
        writer->min_max_size = sizeof(min_v);
    } else {
        double cur_min, cur_max;
        memcpy(&cur_min, writer->min_value, sizeof(cur_min));
        memcpy(&cur_max, writer->max_value, sizeof(cur_max));
        if (cur_min < min_v) min_v = cur_min;
        if (cur_max > max_v) max_v = cur_max;
    }
    memcpy(writer->min_value, &min_v, sizeof(min_v));
    memcpy(writer->max_value, &max_v, sizeof(max_v));
    return CARQUET_OK;
#else
    carquet_status_t status = carquet_encode_plain_double(values, count, &writer->values_buffer);
    if (status == CARQUET_OK) {
        update_statistics_double(writer, values, count);
    }
    return status;
#endif
}

static carquet_status_t encode_float_values(
    carquet_page_writer_t* writer,
    const float* values,
    int64_t count) {

    if (count == 0) {
        return CARQUET_OK;
    }

    if (writer->encoding == CARQUET_ENCODING_BYTE_STREAM_SPLIT) {
        size_t bytes_needed = (size_t)count * sizeof(float);
        size_t offset = writer->values_buffer.size;
        uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, bytes_needed);
        size_t bytes_written = 0;
        if (!dest) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        carquet_status_t status = carquet_byte_stream_split_encode_float(
            values, count, dest, bytes_needed, &bytes_written);
        if (status != CARQUET_OK || bytes_written != bytes_needed) {
            writer->values_buffer.size = offset;
            return status != CARQUET_OK ? status : CARQUET_ERROR_ENCODE;
        }
        return CARQUET_OK;
    }

    return carquet_encode_plain_float(values, count, &writer->values_buffer);
}

static carquet_status_t encode_double_values(
    carquet_page_writer_t* writer,
    const double* values,
    int64_t count) {

    if (count == 0) {
        return CARQUET_OK;
    }

    if (writer->encoding == CARQUET_ENCODING_BYTE_STREAM_SPLIT) {
        size_t bytes_needed = (size_t)count * sizeof(double);
        size_t offset = writer->values_buffer.size;
        uint8_t* dest = carquet_buffer_advance(&writer->values_buffer, bytes_needed);
        size_t bytes_written = 0;
        if (!dest) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        carquet_status_t status = carquet_byte_stream_split_encode_double(
            values, count, dest, bytes_needed, &bytes_written);
        if (status != CARQUET_OK || bytes_written != bytes_needed) {
            writer->values_buffer.size = offset;
            return status != CARQUET_OK ? status : CARQUET_ERROR_ENCODE;
        }
        return CARQUET_OK;
    }

    return carquet_encode_plain_double(values, count, &writer->values_buffer);
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
        num_non_null = carquet_dispatch_count_non_nulls(def_levels, num_values,
                                                        writer->max_def_level);
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
            status = writer->write_statistics
                ? encode_plain_i32_with_stats(writer, ints, num_non_null)
                : carquet_encode_plain_int32(ints, num_non_null, &writer->values_buffer);
            break;
        }

        case CARQUET_PHYSICAL_INT64: {
            const int64_t* ints = (const int64_t*)values;
            status = writer->write_statistics
                ? encode_plain_i64_with_stats(writer, ints, num_non_null)
                : carquet_encode_plain_int64(ints, num_non_null, &writer->values_buffer);
            break;
        }

        case CARQUET_PHYSICAL_FLOAT: {
            const float* floats = (const float*)values;
            if (writer->encoding == CARQUET_ENCODING_BYTE_STREAM_SPLIT || !writer->write_statistics) {
                status = encode_float_values(writer, floats, num_non_null);
                if (status == CARQUET_OK && writer->write_statistics) {
                    update_statistics_float(writer, floats, num_non_null);
                }
            } else {
                status = encode_plain_float_with_stats(writer, floats, num_non_null);
            }
            break;
        }

        case CARQUET_PHYSICAL_DOUBLE: {
            const double* doubles = (const double*)values;
            if (writer->encoding == CARQUET_ENCODING_BYTE_STREAM_SPLIT || !writer->write_statistics) {
                status = encode_double_values(writer, doubles, num_non_null);
                if (status == CARQUET_OK && writer->write_statistics) {
                    update_statistics_double(writer, doubles, num_non_null);
                }
            } else {
                status = encode_plain_double_with_stats(writer, doubles, num_non_null);
            }
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
        carquet_status_t reserve_status = carquet_buffer_reserve(temp_buffer, bound);
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

static carquet_status_t build_page_payload(
    carquet_page_writer_t* writer,
    const uint8_t** payload_data,
    size_t* payload_size) {

    size_t total_size = writer->rep_levels_buffer.size +
                        writer->def_levels_buffer.size +
                        writer->values_buffer.size;

    if (writer->rep_levels_buffer.size == 0 && writer->def_levels_buffer.size == 0) {
        *payload_data = writer->values_buffer.data;
        *payload_size = writer->values_buffer.size;
        return CARQUET_OK;
    }

    carquet_buffer_clear(&writer->staging_buffer);
    carquet_status_t status = carquet_buffer_reserve(&writer->staging_buffer, total_size);
    if (status != CARQUET_OK) {
        return status;
    }

    if (writer->rep_levels_buffer.size > 0) {
        status = carquet_buffer_append(&writer->staging_buffer,
                                       writer->rep_levels_buffer.data,
                                       writer->rep_levels_buffer.size);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    if (writer->def_levels_buffer.size > 0) {
        status = carquet_buffer_append(&writer->staging_buffer,
                                       writer->def_levels_buffer.data,
                                       writer->def_levels_buffer.size);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    if (writer->values_buffer.size > 0) {
        status = carquet_buffer_append(&writer->staging_buffer,
                                       writer->values_buffer.data,
                                       writer->values_buffer.size);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    *payload_data = writer->staging_buffer.data;
    *payload_size = writer->staging_buffer.size;
    return CARQUET_OK;
}

static uint32_t compute_page_crc(
    const carquet_page_writer_t* writer,
    const uint8_t* payload_data,
    size_t payload_size) {

    if (!writer->write_crc) {
        return 0;
    }

    if (payload_data) {
        return carquet_crc32(payload_data, payload_size);
    }

    uint32_t crc = 0;
    if (writer->rep_levels_buffer.size > 0) {
        crc = carquet_crc32_update(crc,
                                   writer->rep_levels_buffer.data,
                                   writer->rep_levels_buffer.size);
    }
    if (writer->def_levels_buffer.size > 0) {
        crc = carquet_crc32_update(crc,
                                   writer->def_levels_buffer.data,
                                   writer->def_levels_buffer.size);
    }
    if (writer->values_buffer.size > 0) {
        crc = carquet_crc32_update(crc,
                                   writer->values_buffer.data,
                                   writer->values_buffer.size);
    }
    return crc;
}

static carquet_status_t append_data_page_header(
    carquet_buffer_t* output_buffer,
    const carquet_page_writer_t* writer,
    int32_t uncompressed_size,
    int32_t compressed_size,
    uint32_t page_crc) {

    carquet_status_t status = carquet_buffer_reserve(
        output_buffer, output_buffer->size + 128);
    if (status != CARQUET_OK) {
        return status;
    }

    thrift_encoder_t enc;
    thrift_encoder_init(&enc, output_buffer);

    thrift_write_struct_begin(&enc);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
    thrift_write_i32(&enc, CARQUET_PAGE_DATA);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
    thrift_write_i32(&enc, uncompressed_size);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 3);
    thrift_write_i32(&enc, compressed_size);

    if (writer->write_crc) {
        thrift_write_field_header(&enc, THRIFT_TYPE_I32, 4);
        thrift_write_i32(&enc, (int32_t)page_crc);
    }

    thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 5);
    thrift_write_struct_begin(&enc);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
    thrift_write_i32(&enc, (int32_t)writer->num_values);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 2);
    thrift_write_i32(&enc, (int32_t)writer->encoding);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 3);
    thrift_write_i32(&enc, CARQUET_ENCODING_RLE);
    thrift_write_field_header(&enc, THRIFT_TYPE_I32, 4);
    thrift_write_i32(&enc, CARQUET_ENCODING_RLE);

    if (writer->write_statistics && writer->has_min_max) {
        thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 5);
        thrift_write_struct_begin(&enc);
        thrift_write_field_header(&enc, THRIFT_TYPE_I64, 3);
        thrift_write_i64(&enc, writer->num_nulls);
        thrift_write_field_header(&enc, THRIFT_TYPE_BINARY, 5);
        thrift_write_binary(&enc, writer->max_value, (int32_t)writer->min_max_size);
        thrift_write_field_header(&enc, THRIFT_TYPE_BINARY, 6);
        thrift_write_binary(&enc, writer->min_value, (int32_t)writer->min_max_size);
        thrift_write_struct_end(&enc);
    }

    thrift_write_struct_end(&enc);
    thrift_write_struct_end(&enc);
    return enc.status;
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
    carquet_status_t status = carquet_page_writer_finalize_to_buffer(
        writer, &writer->page_buffer, page_size, uncompressed_size, compressed_size);
    if (status != CARQUET_OK) {
        return status;
    }

    *page_data = writer->page_buffer.data;
    return CARQUET_OK;
}

carquet_status_t carquet_page_writer_finalize_to_buffer(
    carquet_page_writer_t* writer,
    carquet_buffer_t* output_buffer,
    size_t* page_size,
    int32_t* uncompressed_size,
    int32_t* compressed_size) {

    if (!writer || !output_buffer || !page_size) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    bool has_levels = (writer->rep_levels_buffer.size > 0 ||
                       writer->def_levels_buffer.size > 0);
    const uint8_t* payload_data = NULL;
    size_t payload_size = 0;
    const uint8_t* compressed_data = NULL;
    size_t compressed_data_size = 0;
    carquet_status_t status;

    size_t page_start = output_buffer->size;

    if (writer->compression == CARQUET_COMPRESSION_UNCOMPRESSED) {
        *uncompressed_size = (int32_t)(writer->rep_levels_buffer.size +
                                       writer->def_levels_buffer.size +
                                       writer->values_buffer.size);
        *compressed_size = *uncompressed_size;

        uint32_t page_crc = compute_page_crc(writer, NULL, 0);
        status = carquet_buffer_reserve(output_buffer, output_buffer->size + 128 +
                                        (size_t)*compressed_size);
        if (status != CARQUET_OK) {
            output_buffer->size = page_start;
            return status;
        }

        status = append_data_page_header(output_buffer, writer,
                                         *uncompressed_size, *compressed_size, page_crc);
        if (status != CARQUET_OK) {
            output_buffer->size = page_start;
            return status;
        }

        if (has_levels) {
            if (writer->rep_levels_buffer.size > 0) {
                status = carquet_buffer_append(output_buffer,
                                              writer->rep_levels_buffer.data,
                                              writer->rep_levels_buffer.size);
                if (status != CARQUET_OK) {
                    output_buffer->size = page_start;
                    return status;
                }
            }
            if (writer->def_levels_buffer.size > 0) {
                status = carquet_buffer_append(output_buffer,
                                              writer->def_levels_buffer.data,
                                              writer->def_levels_buffer.size);
                if (status != CARQUET_OK) {
                    output_buffer->size = page_start;
                    return status;
                }
            }
        }

        status = carquet_buffer_append(output_buffer,
                                       writer->values_buffer.data,
                                       writer->values_buffer.size);
        if (status != CARQUET_OK) {
            output_buffer->size = page_start;
            return status;
        }

        *page_size = output_buffer->size - page_start;
        return CARQUET_OK;
    }

    status = build_page_payload(writer, &payload_data, &payload_size);
    if (status != CARQUET_OK) {
        return status;
    }

    *uncompressed_size = (int32_t)payload_size;
    status = compress_data(writer->compression,
                           payload_data, payload_size,
                           &writer->compress_buffer,
                           &compressed_data,
                           &compressed_data_size,
                           writer->compression_level);
    if (status != CARQUET_OK) {
        return status;
    }

    *compressed_size = (int32_t)compressed_data_size;
    status = carquet_buffer_reserve(output_buffer, output_buffer->size + 128 +
                                    compressed_data_size);
    if (status != CARQUET_OK) {
        output_buffer->size = page_start;
        return status;
    }

    status = append_data_page_header(output_buffer, writer,
                                     *uncompressed_size, *compressed_size,
                                     compute_page_crc(writer, compressed_data, compressed_data_size));
    if (status != CARQUET_OK) {
        output_buffer->size = page_start;
        return status;
    }

    status = carquet_buffer_append(output_buffer, compressed_data, compressed_data_size);
    if (status != CARQUET_OK) {
        output_buffer->size = page_start;
        return status;
    }

    *page_size = output_buffer->size - page_start;
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
