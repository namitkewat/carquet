/**
 * @file column_writer.c
 * @brief Column chunk writing implementation
 *
 * Manages writing values to a column chunk, handling page breaks,
 * dictionary encoding, and column-level metadata.
 */

#include <carquet/carquet.h>
#include <carquet/error.h>
#include "core/buffer.h"
#include "thrift/thrift_encode.h"
#include "thrift/parquet_types.h"
#include <stdlib.h>
#include <string.h>

/* Forward declarations for bloom filter and page index */
typedef struct carquet_bloom_filter carquet_bloom_filter_t;
typedef struct carquet_column_index_builder carquet_column_index_builder_t;
typedef struct carquet_offset_index_builder carquet_offset_index_builder_t;

extern carquet_bloom_filter_t* carquet_bloom_filter_create_with_ndv(int64_t ndv, double fpp);
extern void carquet_bloom_filter_destroy(carquet_bloom_filter_t* filter);
extern void carquet_bloom_filter_insert_i32(carquet_bloom_filter_t* filter, int32_t value);
extern void carquet_bloom_filter_insert_i64(carquet_bloom_filter_t* filter, int64_t value);
extern void carquet_bloom_filter_insert_float(carquet_bloom_filter_t* filter, float value);
extern void carquet_bloom_filter_insert_double(carquet_bloom_filter_t* filter, double value);
extern void carquet_bloom_filter_insert_bytes(carquet_bloom_filter_t* filter,
                                               const uint8_t* data, size_t len);
extern const uint8_t* carquet_bloom_filter_data(const carquet_bloom_filter_t* filter);
extern size_t carquet_bloom_filter_size(const carquet_bloom_filter_t* filter);

extern carquet_column_index_builder_t* carquet_column_index_builder_create(
    carquet_physical_type_t type, int32_t type_length);
extern void carquet_column_index_builder_destroy(carquet_column_index_builder_t* builder);
extern carquet_status_t carquet_column_index_add_page(
    carquet_column_index_builder_t* builder,
    int64_t null_count, const void* min_value, int32_t min_value_len,
    const void* max_value, int32_t max_value_len, bool is_null_page);
extern carquet_status_t carquet_column_index_serialize(
    const carquet_column_index_builder_t* builder, carquet_buffer_t* output);

extern carquet_offset_index_builder_t* carquet_offset_index_builder_create(bool track_uncompressed);
extern void carquet_offset_index_builder_destroy(carquet_offset_index_builder_t* builder);
extern carquet_status_t carquet_offset_index_add_page(
    carquet_offset_index_builder_t* builder,
    int64_t offset, int32_t compressed_size,
    int64_t first_row_index, int32_t uncompressed_size);
extern carquet_status_t carquet_offset_index_serialize(
    const carquet_offset_index_builder_t* builder, carquet_buffer_t* output);

/* Forward declaration from page_writer.c */
typedef struct carquet_page_writer carquet_page_writer_t;

extern carquet_page_writer_t* carquet_page_writer_create(
    carquet_physical_type_t type,
    carquet_encoding_t encoding,
    carquet_compression_t compression,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length,
    int32_t compression_level);

extern void carquet_page_writer_destroy(carquet_page_writer_t* writer);
extern void carquet_page_writer_reset(carquet_page_writer_t* writer);

extern carquet_status_t carquet_page_writer_add_values(
    carquet_page_writer_t* writer,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels);

extern carquet_status_t carquet_page_writer_finalize(
    carquet_page_writer_t* writer,
    const uint8_t** page_data,
    size_t* page_size,
    int32_t* uncompressed_size,
    int32_t* compressed_size);
extern carquet_status_t carquet_page_writer_finalize_to_buffer(
    carquet_page_writer_t* writer,
    carquet_buffer_t* output_buffer,
    size_t* page_size,
    int32_t* uncompressed_size,
    int32_t* compressed_size);

extern size_t carquet_page_writer_estimated_size(const carquet_page_writer_t* writer);
extern int64_t carquet_page_writer_num_values(const carquet_page_writer_t* writer);
extern void carquet_page_writer_set_crc(carquet_page_writer_t* writer, bool enabled);
extern void carquet_page_writer_set_statistics(carquet_page_writer_t* writer, bool enabled);

/* ============================================================================
 * Column Writer Structure
 * ============================================================================
 */

typedef struct carquet_column_writer_internal {
    carquet_page_writer_t* page_writer;
    carquet_buffer_t column_buffer;  /* All pages for this column chunk */

    /* Column configuration */
    carquet_physical_type_t type;
    carquet_encoding_t encoding;
    carquet_compression_t compression;
    int32_t type_length;
    int16_t max_def_level;
    int16_t max_rep_level;

    /* Page size limits */
    size_t target_page_size;
    size_t max_page_size;

    /* Statistics */
    int64_t total_values;
    int64_t total_nulls;
    int64_t total_uncompressed_size;
    int64_t total_compressed_size;
    int32_t num_pages;

    /* Min/max tracking */
    bool has_min_max;
    uint8_t min_value[64];
    uint8_t max_value[64];
    size_t min_max_size;

    /* Column path for metadata */
    char** path_in_schema;
    int path_depth;

    /* Bloom filter (optional) */
    carquet_bloom_filter_t* bloom_filter;
    int64_t bloom_ndv;

    /* Page index builders (optional) */
    carquet_column_index_builder_t* column_index;
    carquet_offset_index_builder_t* offset_index;
    bool page_index_enabled;
    int64_t page_row_offset;      /* Row offset for current page (for offset index) */
    int64_t column_file_offset;   /* File offset where this column starts */
} carquet_column_writer_internal_t;

/* ============================================================================
 * Column Writer Lifecycle
 * ============================================================================
 */

carquet_column_writer_internal_t* carquet_column_writer_create(
    carquet_physical_type_t type,
    carquet_encoding_t encoding,
    carquet_compression_t compression,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length,
    size_t target_page_size,
    int32_t compression_level) {

    carquet_column_writer_internal_t* writer = calloc(1, sizeof(*writer));
    if (!writer) return NULL;

    writer->page_writer = carquet_page_writer_create(
        type, encoding, compression, max_def_level, max_rep_level, type_length,
        compression_level);

    if (!writer->page_writer) {
        free(writer);
        return NULL;
    }

    carquet_buffer_init(&writer->column_buffer);

    writer->type = type;
    writer->encoding = encoding;
    writer->compression = compression;
    writer->type_length = type_length;
    writer->max_def_level = max_def_level;
    writer->max_rep_level = max_rep_level;
    writer->target_page_size = target_page_size > 0 ? target_page_size : (1024 * 1024);
    writer->max_page_size = writer->target_page_size * 2;

    return writer;
}

void carquet_column_writer_destroy(carquet_column_writer_internal_t* writer) {
    if (writer) {
        if (writer->page_writer) {
            carquet_page_writer_destroy(writer->page_writer);
        }
        carquet_buffer_destroy(&writer->column_buffer);

        /* Free path strings */
        if (writer->path_in_schema) {
            for (int i = 0; i < writer->path_depth; i++) {
                free(writer->path_in_schema[i]);
            }
            free(writer->path_in_schema);
        }

        if (writer->bloom_filter) {
            carquet_bloom_filter_destroy(writer->bloom_filter);
        }
        if (writer->column_index) {
            carquet_column_index_builder_destroy(writer->column_index);
        }
        if (writer->offset_index) {
            carquet_offset_index_builder_destroy(writer->offset_index);
        }

        free(writer);
    }
}

void carquet_column_writer_set_crc(carquet_column_writer_internal_t* writer, bool enabled) {
    if (writer) {
        carquet_page_writer_set_crc(writer->page_writer, enabled);
    }
}

void carquet_column_writer_reset(carquet_column_writer_internal_t* writer) {
    if (!writer) return;

    carquet_page_writer_reset(writer->page_writer);
    carquet_buffer_clear(&writer->column_buffer);

    writer->total_values = 0;
    writer->total_nulls = 0;
    writer->total_uncompressed_size = 0;
    writer->total_compressed_size = 0;
    writer->num_pages = 0;
    writer->has_min_max = false;
    writer->min_max_size = 0;
    writer->page_row_offset = 0;
    writer->column_file_offset = 0;

    if (writer->bloom_filter) {
        carquet_bloom_filter_destroy(writer->bloom_filter);
        writer->bloom_filter = NULL;
    }
    if (writer->bloom_ndv > 0) {
        writer->bloom_filter = carquet_bloom_filter_create_with_ndv(writer->bloom_ndv, 0.01);
    }

    if (writer->column_index) {
        carquet_column_index_builder_destroy(writer->column_index);
        writer->column_index = NULL;
    }
    if (writer->offset_index) {
        carquet_offset_index_builder_destroy(writer->offset_index);
        writer->offset_index = NULL;
    }
    if (writer->page_index_enabled) {
        writer->column_index = carquet_column_index_builder_create(writer->type, writer->type_length);
        writer->offset_index = carquet_offset_index_builder_create(true);
    }
}

/* ============================================================================
 * Page Flushing
 * ============================================================================
 */

/* Forward declarations for page writer statistics */
extern bool carquet_page_writer_get_statistics(
    const carquet_page_writer_t* writer,
    const uint8_t** min_value, const uint8_t** max_value,
    size_t* value_size, int64_t* null_count);
extern int64_t carquet_page_writer_null_count(const carquet_page_writer_t* writer);

static carquet_status_t flush_current_page(carquet_column_writer_internal_t* writer) {
    if (carquet_page_writer_num_values(writer->page_writer) == 0) {
        return CARQUET_OK;
    }

    size_t page_size;
    int32_t uncompressed_size;
    int32_t compressed_size;

    /* Capture per-page statistics before finalize (for page index) */
    const uint8_t* page_min = NULL;
    const uint8_t* page_max = NULL;
    size_t stat_size = 0;
    int64_t page_null_count = 0;
    bool has_stats = false;

    if (writer->column_index) {
        has_stats = carquet_page_writer_get_statistics(
            writer->page_writer, &page_min, &page_max, &stat_size, &page_null_count);
    }

    size_t page_start = writer->column_buffer.size;
    carquet_status_t status = carquet_page_writer_finalize_to_buffer(
        writer->page_writer, &writer->column_buffer, &page_size,
        &uncompressed_size, &compressed_size);

    if (status != CARQUET_OK) {
        return status;
    }

    /* Record page index entries before appending */
    if (writer->column_index) {
        bool is_null_page = !has_stats;
        if (!page_null_count) page_null_count = carquet_page_writer_null_count(writer->page_writer);
        carquet_column_index_add_page(
            writer->column_index,
            page_null_count,
            has_stats ? page_min : NULL, has_stats ? (int32_t)stat_size : 0,
            has_stats ? page_max : NULL, has_stats ? (int32_t)stat_size : 0,
            is_null_page);
    }

    if (writer->offset_index) {
        /* Page offset = column's file offset + current buffer position (before append) */
        int64_t page_offset = writer->column_file_offset + (int64_t)page_start;
        carquet_offset_index_add_page(
            writer->offset_index,
            page_offset,
            (int32_t)page_size,
            writer->page_row_offset,
            uncompressed_size);
        writer->page_row_offset += carquet_page_writer_num_values(writer->page_writer);
    }

    /* Update statistics */
    writer->total_uncompressed_size += uncompressed_size;
    writer->total_compressed_size += compressed_size;
    writer->num_pages++;

    /* Reset page writer for next page */
    carquet_page_writer_reset(writer->page_writer);

    return CARQUET_OK;
}

/* ============================================================================
 * Writing Values
 * ============================================================================
 */

/* Byte size of a value in memory for fixed-size physical types.
 * Returns 0 for variable-length types (BYTE_ARRAY). */
static size_t physical_type_stride(carquet_physical_type_t type, int32_t type_length) {
    switch (type) {
        case CARQUET_PHYSICAL_BOOLEAN:  return 1;
        case CARQUET_PHYSICAL_INT32:    return 4;
        case CARQUET_PHYSICAL_INT64:    return 8;
        case CARQUET_PHYSICAL_FLOAT:    return 4;
        case CARQUET_PHYSICAL_DOUBLE:   return 8;
        case CARQUET_PHYSICAL_INT96:    return 12;
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY: return (size_t)type_length;
        default: return 0;
    }
}

static void bloom_filter_insert_chunk(
    carquet_column_writer_internal_t* writer,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels) {

    if (!writer->bloom_filter) return;

    int64_t num_non_null = num_values;
    if (def_levels && writer->max_def_level > 0) {
        num_non_null = 0;
        for (int64_t i = 0; i < num_values; i++) {
            if (def_levels[i] == writer->max_def_level) num_non_null++;
        }
    }

    switch (writer->type) {
        case CARQUET_PHYSICAL_INT32: {
            const int32_t* v = (const int32_t*)values;
            for (int64_t i = 0; i < num_non_null; i++)
                carquet_bloom_filter_insert_i32(writer->bloom_filter, v[i]);
            break;
        }
        case CARQUET_PHYSICAL_INT64: {
            const int64_t* v = (const int64_t*)values;
            for (int64_t i = 0; i < num_non_null; i++)
                carquet_bloom_filter_insert_i64(writer->bloom_filter, v[i]);
            break;
        }
        case CARQUET_PHYSICAL_FLOAT: {
            const float* v = (const float*)values;
            for (int64_t i = 0; i < num_non_null; i++)
                carquet_bloom_filter_insert_float(writer->bloom_filter, v[i]);
            break;
        }
        case CARQUET_PHYSICAL_DOUBLE: {
            const double* v = (const double*)values;
            for (int64_t i = 0; i < num_non_null; i++)
                carquet_bloom_filter_insert_double(writer->bloom_filter, v[i]);
            break;
        }
        case CARQUET_PHYSICAL_BYTE_ARRAY: {
            const carquet_byte_array_t* v = (const carquet_byte_array_t*)values;
            for (int64_t i = 0; i < num_non_null; i++)
                carquet_bloom_filter_insert_bytes(writer->bloom_filter, v[i].data, v[i].length);
            break;
        }
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY: {
            const uint8_t* v = (const uint8_t*)values;
            for (int64_t i = 0; i < num_non_null; i++)
                carquet_bloom_filter_insert_bytes(writer->bloom_filter,
                    v + i * writer->type_length, writer->type_length);
            break;
        }
        default:
            break;
    }
}

carquet_status_t carquet_column_writer_write_batch(
    carquet_column_writer_internal_t* writer,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels) {

    if (!writer || !values) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* For fixed-size types, split large batches into page-sized chunks.
     * This keeps the working set in cache and avoids huge buffer
     * reallocations that would otherwise occur when accumulating
     * hundreds of MB into a single page. */
    size_t stride = physical_type_stride(writer->type, writer->type_length);

    int64_t max_chunk = num_values;
    if (stride > 0) {
        max_chunk = (int64_t)(writer->target_page_size / stride);
        if (max_chunk < 1024) max_chunk = 1024;

        /* Pre-allocate column buffer to avoid repeated realloc+copy as
         * pages accumulate. Each page adds ~target_page_size + header. */
        if (num_values > max_chunk) {
            size_t expected = (size_t)num_values * stride;
            /* Add ~2% overhead for page headers */
            expected += expected / 50;
            carquet_buffer_reserve(&writer->column_buffer, expected);
        }
    }

    const uint8_t* val_bytes = (const uint8_t*)values;
    int64_t offset = 0;

    while (offset < num_values) {
        int64_t chunk = num_values - offset;
        if (chunk > max_chunk) chunk = max_chunk;

        const void* chunk_values = (stride > 0)
            ? (const void*)(val_bytes + offset * stride)
            : (const void*)((const carquet_byte_array_t*)values + offset);

        carquet_status_t status = carquet_page_writer_add_values(
            writer->page_writer, chunk_values, chunk,
            def_levels ? def_levels + offset : NULL,
            rep_levels ? rep_levels + offset : NULL);

        if (status != CARQUET_OK) return status;

        writer->total_values += chunk;

        bloom_filter_insert_chunk(writer, chunk_values, chunk,
                                  def_levels ? def_levels + offset : NULL);

        offset += chunk;

        /* Flush page when it reaches target size */
        if (carquet_page_writer_estimated_size(writer->page_writer) >= writer->target_page_size) {
            status = flush_current_page(writer);
            if (status != CARQUET_OK) return status;
        }
    }

    return CARQUET_OK;
}

/* ============================================================================
 * Finalization
 * ============================================================================
 */

carquet_status_t carquet_column_writer_finalize(
    carquet_column_writer_internal_t* writer,
    const uint8_t** data,
    size_t* size,
    int64_t* total_values,
    int64_t* total_compressed_size,
    int64_t* total_uncompressed_size) {

    if (!writer) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Flush any remaining data */
    carquet_status_t status = flush_current_page(writer);
    if (status != CARQUET_OK) {
        return status;
    }

    if (data) *data = writer->column_buffer.data;
    if (size) *size = writer->column_buffer.size;
    if (total_values) *total_values = writer->total_values;
    if (total_compressed_size) *total_compressed_size = writer->total_compressed_size;
    if (total_uncompressed_size) *total_uncompressed_size = writer->total_uncompressed_size;

    return CARQUET_OK;
}

void carquet_column_writer_set_statistics(
    carquet_column_writer_internal_t* writer,
    bool enabled) {
    if (writer && writer->page_writer) {
        carquet_page_writer_set_statistics(writer->page_writer, enabled);
    }
}

int64_t carquet_column_writer_num_values(const carquet_column_writer_internal_t* writer) {
    return writer ? writer->total_values : 0;
}

int32_t carquet_column_writer_num_pages(const carquet_column_writer_internal_t* writer) {
    return writer ? writer->num_pages : 0;
}

void carquet_column_writer_enable_bloom_filter(
    carquet_column_writer_internal_t* writer, int64_t ndv) {
    if (!writer || writer->bloom_filter) return;
    writer->bloom_ndv = ndv > 0 ? ndv : 100000;
    writer->bloom_filter = carquet_bloom_filter_create_with_ndv(
        writer->bloom_ndv, 0.01);
}

void carquet_column_writer_enable_page_index(
    carquet_column_writer_internal_t* writer) {
    if (!writer) return;
    writer->page_index_enabled = true;
    if (!writer->column_index) {
        writer->column_index = carquet_column_index_builder_create(
            writer->type, writer->type_length);
    }
    if (!writer->offset_index) {
        writer->offset_index = carquet_offset_index_builder_create(true);
    }
}

void carquet_column_writer_set_file_offset(
    carquet_column_writer_internal_t* writer, int64_t offset) {
    if (writer) writer->column_file_offset = offset;
}

carquet_bloom_filter_t* carquet_column_writer_get_bloom_filter(
    const carquet_column_writer_internal_t* writer) {
    return writer ? writer->bloom_filter : NULL;
}

carquet_column_index_builder_t* carquet_column_writer_get_column_index(
    const carquet_column_writer_internal_t* writer) {
    return writer ? writer->column_index : NULL;
}

carquet_offset_index_builder_t* carquet_column_writer_get_offset_index(
    const carquet_column_writer_internal_t* writer) {
    return writer ? writer->offset_index : NULL;
}
