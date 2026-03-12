/**
 * @file file_writer.c
 * @brief Parquet file writing implementation
 *
 * Manages writing a complete Parquet file including:
 * - File header (PAR1 magic)
 * - Row groups via row_group_writer
 * - File metadata serialization
 * - Footer with metadata size and PAR1 magic
 */

#include <carquet/carquet.h>
#include <carquet/error.h>
#include "core/buffer.h"
#include "core/arena.h"
#include "reader/reader_internal.h"
#include "thrift/thrift_encode.h"
#include "thrift/parquet_types.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* Parquet magic bytes */
static const uint8_t PARQUET_MAGIC[4] = {'P', 'A', 'R', '1'};

/* Forward declaration from row_group_writer.c */
typedef struct carquet_row_group_writer carquet_row_group_writer_t;

typedef struct column_chunk_info {
    int64_t file_offset;
    int64_t total_compressed_size;
    int64_t total_uncompressed_size;
    int64_t num_values;
    carquet_physical_type_t type;
    carquet_encoding_t encoding;
    carquet_compression_t compression;
    int32_t type_length;
    char* path;
} column_chunk_info_t;

extern carquet_row_group_writer_t* carquet_row_group_writer_create(
    const carquet_schema_t* schema,
    carquet_compression_t compression,
    size_t target_page_size,
    int64_t file_offset);

extern void carquet_row_group_writer_destroy(carquet_row_group_writer_t* writer);
extern void carquet_row_group_writer_reset(
    carquet_row_group_writer_t* writer,
    int64_t file_offset);

extern carquet_status_t carquet_row_group_writer_add_column(
    carquet_row_group_writer_t* writer,
    const char* name,
    carquet_physical_type_t type,
    int16_t max_def_level,
    int16_t max_rep_level,
    int32_t type_length);

extern carquet_status_t carquet_row_group_writer_write_column(
    carquet_row_group_writer_t* writer,
    int column_index,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels);

extern carquet_status_t carquet_row_group_writer_finalize(
    carquet_row_group_writer_t* writer,
    const uint8_t** data,
    size_t* size,
    int64_t num_rows);

extern carquet_status_t carquet_row_group_writer_write_to_file(
    carquet_row_group_writer_t* writer,
    FILE* file,
    size_t* total_size,
    int64_t num_rows);

extern int carquet_row_group_writer_num_columns(const carquet_row_group_writer_t* writer);
extern int64_t carquet_row_group_writer_num_rows(const carquet_row_group_writer_t* writer);
extern int64_t carquet_row_group_writer_total_byte_size(const carquet_row_group_writer_t* writer);
extern const column_chunk_info_t* carquet_row_group_writer_get_column_info(
    const carquet_row_group_writer_t* writer, int index);

extern void carquet_row_group_writer_set_options(
    carquet_row_group_writer_t* writer,
    bool write_bloom_filters, bool write_page_index,
    bool write_statistics,
    bool write_crc,
    int32_t compression_level);

/* Bloom filter and page index accessors */
typedef struct carquet_bloom_filter carquet_bloom_filter_t;
typedef struct carquet_column_index_builder carquet_column_index_builder_t;
typedef struct carquet_offset_index_builder carquet_offset_index_builder_t;

extern carquet_bloom_filter_t* carquet_row_group_writer_get_bloom_filter(
    const carquet_row_group_writer_t* writer, int index);
extern carquet_column_index_builder_t* carquet_row_group_writer_get_column_index(
    const carquet_row_group_writer_t* writer, int index);
extern carquet_offset_index_builder_t* carquet_row_group_writer_get_offset_index(
    const carquet_row_group_writer_t* writer, int index);

extern const uint8_t* carquet_bloom_filter_data(const carquet_bloom_filter_t* filter);
extern size_t carquet_bloom_filter_size(const carquet_bloom_filter_t* filter);
extern carquet_status_t carquet_column_index_serialize(
    const carquet_column_index_builder_t* builder, carquet_buffer_t* output);
extern carquet_status_t carquet_offset_index_serialize(
    const carquet_offset_index_builder_t* builder, carquet_buffer_t* output);

/* ============================================================================
 * Writer Schema Structure (for building)
 * ============================================================================
 */

typedef struct writer_column_def {
    char* name;
    carquet_physical_type_t physical_type;
    carquet_logical_type_t logical_type;
    carquet_field_repetition_t repetition;
    int32_t type_length;
    int16_t max_def_level;
    int16_t max_rep_level;
} writer_column_def_t;

/* ============================================================================
 * Row Group Metadata Storage
 * ============================================================================
 */

typedef struct row_group_column_info {
    int64_t file_offset;
    int64_t total_compressed_size;
    int64_t total_uncompressed_size;
    int64_t num_values;
    carquet_physical_type_t type;
    carquet_compression_t codec;
    int64_t data_page_offset;
    bool has_bloom_filter_offset;
    int64_t bloom_filter_offset;
    bool has_bloom_filter_length;
    int32_t bloom_filter_length;
    bool has_column_index_offset;
    int64_t column_index_offset;
    bool has_column_index_length;
    int32_t column_index_length;
    bool has_offset_index_offset;
    int64_t offset_index_offset;
    bool has_offset_index_length;
    int32_t offset_index_length;
} row_group_column_info_t;

typedef struct row_group_info {
    int64_t file_offset;
    int64_t num_rows;
    int64_t total_byte_size;
    int64_t total_compressed_size;
    int16_t ordinal;
    row_group_column_info_t* columns;
    int32_t num_columns;
} row_group_info_t;

/* ============================================================================
 * Writer Structure
 * ============================================================================
 */

struct carquet_writer {
    FILE* file;
    bool owns_file;
    char* path;

    /* Schema */
    writer_column_def_t* columns;
    int32_t num_columns;
    int32_t column_capacity;

    /* Full schema elements (including groups) for metadata serialization */
    parquet_schema_element_t* schema_elements;
    int32_t num_schema_elements;
    char*** column_paths;
    int32_t* column_path_lens;
    carquet_encoding_t (*column_encodings)[2];

    /* Options */
    carquet_writer_options_t options;

    /* Current row group */
    carquet_row_group_writer_t* current_row_group;
    int64_t current_row_group_rows;
    int64_t* column_values_written;  /* Values written per column in current row group */

    /* Completed row groups */
    row_group_info_t* row_groups;
    int32_t num_row_groups;
    int32_t row_groups_capacity;

    /* File state */
    int64_t file_offset;
    int64_t total_rows;
    bool header_written;

    /* Arena for metadata allocations */
    carquet_arena_t arena;
};

/* ============================================================================
 * Writer Options
 * ============================================================================
 */

void carquet_writer_options_init(carquet_writer_options_t* options) {
    /* options is nonnull per API contract */
    memset(options, 0, sizeof(*options));
    options->compression = CARQUET_COMPRESSION_UNCOMPRESSED;
    options->compression_level = 0;
    options->row_group_size = 128 * 1024 * 1024;  /* 128 MB */
    options->page_size = 1024 * 1024;              /* 1 MB */
    options->write_statistics = true;
    options->write_crc = true;
    options->write_page_index = false;
    options->write_bloom_filters = false;
    options->dictionary_encoding = CARQUET_ENCODING_PLAIN_DICTIONARY;
    options->dictionary_page_size = 1024 * 1024;   /* 1 MB */
    options->created_by = "Carquet";
}

/* ============================================================================
 * Internal Helpers
 * ============================================================================
 */

static carquet_status_t write_magic(FILE* file) {
    if (fwrite(PARQUET_MAGIC, 1, 4, file) != 4) {
        return CARQUET_ERROR_FILE_WRITE;
    }
    return CARQUET_OK;
}

static carquet_status_t ensure_header_written(carquet_writer_t* writer) {
    if (writer->header_written) {
        return CARQUET_OK;
    }

    carquet_status_t status = write_magic(writer->file);
    if (status != CARQUET_OK) {
        return status;
    }

    writer->file_offset = 4;  /* PAR1 magic */
    writer->header_written = true;
    return CARQUET_OK;
}

static carquet_status_t add_column_internal(
    carquet_writer_t* writer,
    const char* name,
    carquet_physical_type_t physical_type,
    const carquet_logical_type_t* logical_type,
    carquet_field_repetition_t repetition,
    int32_t type_length,
    int16_t max_def_level,
    int16_t max_rep_level) {

    /* Expand capacity if needed */
    if (writer->num_columns >= writer->column_capacity) {
        int32_t new_cap = writer->column_capacity == 0 ? 8 : writer->column_capacity * 2;
        writer_column_def_t* new_cols = realloc(writer->columns,
            new_cap * sizeof(writer_column_def_t));
        if (!new_cols) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        writer->columns = new_cols;

        int64_t* new_values = realloc(writer->column_values_written,
            new_cap * sizeof(int64_t));
        if (!new_values) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        writer->column_values_written = new_values;

        writer->column_capacity = new_cap;
    }

    writer_column_def_t* col = &writer->columns[writer->num_columns];
    memset(col, 0, sizeof(*col));

    col->name = strdup(name);
    if (!col->name) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    col->physical_type = physical_type;
    col->repetition = repetition;
    col->type_length = type_length;

    if (logical_type) {
        col->logical_type = *logical_type;
    }

    col->max_def_level = max_def_level;
    col->max_rep_level = max_rep_level;

    writer->column_values_written[writer->num_columns] = 0;
    writer->num_columns++;

    return CARQUET_OK;
}

/* Store the full schema elements (including groups) for metadata serialization */
static carquet_status_t store_schema_elements(
    carquet_writer_t* writer,
    const carquet_schema_t* schema) {

    writer->num_schema_elements = schema->num_elements;
    writer->schema_elements = calloc(schema->num_elements, sizeof(parquet_schema_element_t));
    if (!writer->schema_elements) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    for (int32_t i = 0; i < schema->num_elements; i++) {
        writer->schema_elements[i] = schema->elements[i];
        if (schema->elements[i].name) {
            writer->schema_elements[i].name = strdup(schema->elements[i].name);
            if (!writer->schema_elements[i].name) {
                return CARQUET_ERROR_OUT_OF_MEMORY;
            }
        }
    }

    return CARQUET_OK;
}

static carquet_status_t build_column_metadata_cache(
    carquet_writer_t* writer,
    const carquet_schema_t* schema) {

    writer->column_paths = calloc((size_t)schema->num_leaves, sizeof(char**));
    writer->column_path_lens = calloc((size_t)schema->num_leaves, sizeof(int32_t));
    writer->column_encodings = calloc((size_t)schema->num_leaves,
        sizeof(*writer->column_encodings));

    if (!writer->column_paths || !writer->column_path_lens || !writer->column_encodings) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    for (int32_t i = 0; i < schema->num_leaves; i++) {
        int32_t elem_idx = schema->leaf_indices[i];
        int32_t depth = 0;

        for (int32_t cur = elem_idx; cur > 0; cur = schema->parent_indices[cur]) {
            depth++;
        }

        if (depth <= 0) {
            depth = 1;
        }

        writer->column_paths[i] = calloc((size_t)depth, sizeof(char*));
        if (!writer->column_paths[i]) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }

        writer->column_path_lens[i] = depth;
        if (depth == 1) {
            writer->column_paths[i][0] = writer->schema_elements[elem_idx].name;
        } else {
            int32_t cur = elem_idx;
            for (int32_t pi = depth - 1; pi >= 0; pi--) {
                writer->column_paths[i][pi] = writer->schema_elements[cur].name;
                cur = schema->parent_indices[cur];
            }
        }

        writer->column_encodings[i][0] =
            (writer->options.compression != CARQUET_COMPRESSION_UNCOMPRESSED &&
             (schema->elements[elem_idx].type == CARQUET_PHYSICAL_FLOAT ||
              schema->elements[elem_idx].type == CARQUET_PHYSICAL_DOUBLE))
                ? CARQUET_ENCODING_BYTE_STREAM_SPLIT
                : CARQUET_ENCODING_PLAIN;
        writer->column_encodings[i][1] = CARQUET_ENCODING_RLE;
    }

    return CARQUET_OK;
}

static carquet_status_t ensure_row_group(carquet_writer_t* writer) {
    if (writer->current_row_group) {
        return CARQUET_OK;
    }

    size_t target_page_size = (size_t)writer->options.page_size;
    if (writer->options.compression == CARQUET_COMPRESSION_ZSTD &&
        target_page_size == 1024 * 1024) {
        target_page_size = 4 * 1024 * 1024;
    }

    writer->current_row_group = carquet_row_group_writer_create(
        NULL,  /* Schema not used directly */
        writer->options.compression,
        target_page_size,
        writer->file_offset);

    if (!writer->current_row_group) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    /* Pass optional feature flags */
    carquet_row_group_writer_set_options(
        writer->current_row_group,
        writer->options.write_bloom_filters,
        writer->options.write_page_index,
        writer->options.write_statistics,
        writer->options.write_crc,
        writer->options.compression_level);

    /* Add all columns to the row group writer */
    for (int32_t i = 0; i < writer->num_columns; i++) {
        writer_column_def_t* col = &writer->columns[i];
        carquet_status_t status = carquet_row_group_writer_add_column(
            writer->current_row_group,
            col->name,
            col->physical_type,
            col->max_def_level,
            col->max_rep_level,
            col->type_length);

        if (status != CARQUET_OK) {
            carquet_row_group_writer_destroy(writer->current_row_group);
            writer->current_row_group = NULL;
            return status;
        }
    }

    writer->current_row_group_rows = 0;
    for (int32_t i = 0; i < writer->num_columns; i++) {
        writer->column_values_written[i] = 0;
    }

    return CARQUET_OK;
}

static carquet_status_t flush_row_group(carquet_writer_t* writer) {
    if (!writer->current_row_group || writer->current_row_group_rows == 0) {
        return CARQUET_OK;
    }

    /* Finalize and write each column directly to file, avoiding
     * an intermediate copy of the entire row group into one buffer */
    size_t size;
    carquet_status_t status = carquet_row_group_writer_write_to_file(
        writer->current_row_group, writer->file, &size,
        writer->current_row_group_rows);

    if (status != CARQUET_OK) {
        return status;
    }

    /* Store row group metadata */
    if (writer->num_row_groups >= writer->row_groups_capacity) {
        int32_t new_cap = writer->row_groups_capacity == 0 ? 4 : writer->row_groups_capacity * 2;
        row_group_info_t* new_rgs = realloc(writer->row_groups,
            new_cap * sizeof(row_group_info_t));
        if (!new_rgs) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        writer->row_groups = new_rgs;
        writer->row_groups_capacity = new_cap;
    }

    row_group_info_t* rg_info = &writer->row_groups[writer->num_row_groups];
    memset(rg_info, 0, sizeof(*rg_info));

    rg_info->file_offset = writer->file_offset;
    rg_info->num_rows = writer->current_row_group_rows;
    rg_info->total_byte_size = carquet_row_group_writer_total_byte_size(writer->current_row_group);
    rg_info->total_compressed_size = (int64_t)size;
    rg_info->ordinal = (int16_t)writer->num_row_groups;

    /* Build column chunks metadata */
    int num_cols = carquet_row_group_writer_num_columns(writer->current_row_group);
    rg_info->num_columns = num_cols;
    rg_info->columns = calloc((size_t)num_cols, sizeof(*rg_info->columns));
    if (!rg_info->columns) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    for (int i = 0; i < num_cols; i++) {
        const column_chunk_info_t* col_info = carquet_row_group_writer_get_column_info(
            writer->current_row_group, i);

        if (!col_info) continue;

        row_group_column_info_t* chunk = &rg_info->columns[i];
        chunk->file_offset = col_info->file_offset;
        chunk->type = col_info->type;
        chunk->codec = col_info->compression;
        chunk->num_values = col_info->num_values;
        chunk->total_compressed_size = col_info->total_compressed_size;
        chunk->total_uncompressed_size = col_info->total_uncompressed_size;
        chunk->data_page_offset = col_info->file_offset;
    }

    writer->num_row_groups++;
    writer->file_offset += (int64_t)size;
    writer->total_rows += writer->current_row_group_rows;

    /* Write bloom filters for each column (after row group data) */
    if (writer->options.write_bloom_filters) {
        for (int i = 0; i < num_cols; i++) {
            carquet_bloom_filter_t* bf = carquet_row_group_writer_get_bloom_filter(
                writer->current_row_group, i);
            if (!bf) continue;

            const uint8_t* bf_data = carquet_bloom_filter_data(bf);
            size_t bf_size = carquet_bloom_filter_size(bf);
            if (!bf_data || bf_size == 0) continue;

            /* Write Bloom Filter Header (Thrift):
             * numBytes: i32, algorithm: MURMUR3_X64_128, hash: XXHASH, compression: UNCOMPRESSED */
            carquet_buffer_t bf_header;
            carquet_buffer_init(&bf_header);
            {
                thrift_encoder_t enc;
                thrift_encoder_init(&enc, &bf_header);
                thrift_write_struct_begin(&enc);
                /* Field 1: numBytes (i32) */
                thrift_write_field_header(&enc, THRIFT_TYPE_I32, 1);
                thrift_write_i32(&enc, (int32_t)bf_size);
                /* Field 2: algorithm (BloomFilterAlgorithm struct) */
                thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 2);
                thrift_write_struct_begin(&enc);
                /* Field 1: SPLIT_BLOCK_BLOOM_FILTER (empty struct) */
                thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 1);
                thrift_write_struct_begin(&enc);
                thrift_write_struct_end(&enc);
                thrift_write_struct_end(&enc);
                /* Field 3: hash (BloomFilterHash struct) */
                thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 3);
                thrift_write_struct_begin(&enc);
                /* Field 1: XXHASH (empty struct) */
                thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 1);
                thrift_write_struct_begin(&enc);
                thrift_write_struct_end(&enc);
                thrift_write_struct_end(&enc);
                /* Field 4: compression (BloomFilterCompression struct) */
                thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 4);
                thrift_write_struct_begin(&enc);
                /* Field 1: UNCOMPRESSED (empty struct) */
                thrift_write_field_header(&enc, THRIFT_TYPE_STRUCT, 1);
                thrift_write_struct_begin(&enc);
                thrift_write_struct_end(&enc);
                thrift_write_struct_end(&enc);
                thrift_write_struct_end(&enc);
            }

            /* Record offset in column metadata */
            row_group_column_info_t* chunk = &rg_info->columns[i];
            chunk->has_bloom_filter_offset = true;
            chunk->bloom_filter_offset = writer->file_offset;
            chunk->has_bloom_filter_length = true;
            chunk->bloom_filter_length = (int32_t)(bf_header.size + bf_size);

            /* Write header + data */
            if (fwrite(bf_header.data, 1, bf_header.size, writer->file) != bf_header.size) {
                carquet_buffer_destroy(&bf_header);
                return CARQUET_ERROR_FILE_WRITE;
            }
            writer->file_offset += (int64_t)bf_header.size;
            carquet_buffer_destroy(&bf_header);

            if (fwrite(bf_data, 1, bf_size, writer->file) != bf_size) {
                return CARQUET_ERROR_FILE_WRITE;
            }
            writer->file_offset += (int64_t)bf_size;
        }
    }

    /* Write column indexes and offset indexes (after bloom filters) */
    if (writer->options.write_page_index) {
        for (int i = 0; i < num_cols; i++) {
            row_group_column_info_t* chunk = &rg_info->columns[i];

            /* Column index */
            carquet_column_index_builder_t* ci = carquet_row_group_writer_get_column_index(
                writer->current_row_group, i);
            if (ci) {
                carquet_buffer_t ci_buf;
                carquet_buffer_init(&ci_buf);
                carquet_column_index_serialize(ci, &ci_buf);
                if (ci_buf.size > 0) {
                    chunk->has_column_index_offset = true;
                    chunk->column_index_offset = writer->file_offset;
                    chunk->has_column_index_length = true;
                    chunk->column_index_length = (int32_t)ci_buf.size;
                    if (fwrite(ci_buf.data, 1, ci_buf.size, writer->file) != ci_buf.size) {
                        carquet_buffer_destroy(&ci_buf);
                        return CARQUET_ERROR_FILE_WRITE;
                    }
                    writer->file_offset += (int64_t)ci_buf.size;
                }
                carquet_buffer_destroy(&ci_buf);
            }

            /* Offset index */
            carquet_offset_index_builder_t* oi = carquet_row_group_writer_get_offset_index(
                writer->current_row_group, i);
            if (oi) {
                carquet_buffer_t oi_buf;
                carquet_buffer_init(&oi_buf);
                carquet_offset_index_serialize(oi, &oi_buf);
                if (oi_buf.size > 0) {
                    chunk->has_offset_index_offset = true;
                    chunk->offset_index_offset = writer->file_offset;
                    chunk->has_offset_index_length = true;
                    chunk->offset_index_length = (int32_t)oi_buf.size;
                    if (fwrite(oi_buf.data, 1, oi_buf.size, writer->file) != oi_buf.size) {
                        carquet_buffer_destroy(&oi_buf);
                        return CARQUET_ERROR_FILE_WRITE;
                    }
                    writer->file_offset += (int64_t)oi_buf.size;
                }
                carquet_buffer_destroy(&oi_buf);
            }
        }
    }

    /* Reuse the current row group writer for the next group. */
    carquet_row_group_writer_reset(writer->current_row_group, writer->file_offset);
    writer->current_row_group_rows = 0;

    return CARQUET_OK;
}

static carquet_status_t build_file_metadata(
    carquet_writer_t* writer,
    parquet_file_metadata_t* metadata) {

    memset(metadata, 0, sizeof(*metadata));

    metadata->version = 2;  /* Parquet version 2 */
    metadata->num_rows = writer->total_rows;
    metadata->created_by = carquet_arena_strdup(&writer->arena,
        writer->options.created_by ? writer->options.created_by : "Carquet");

    /* Build schema from stored elements (includes groups for nested schemas) */
    metadata->num_schema_elements = writer->num_schema_elements;
    metadata->schema = carquet_arena_calloc(&writer->arena, writer->num_schema_elements,
        sizeof(parquet_schema_element_t));

    if (!metadata->schema) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    for (int32_t i = 0; i < writer->num_schema_elements; i++) {
        metadata->schema[i] = writer->schema_elements[i];
        /* Duplicate strings into arena so they outlive the writer */
        if (writer->schema_elements[i].name) {
            metadata->schema[i].name = carquet_arena_strdup(
                &writer->arena, writer->schema_elements[i].name);
        }
    }

    /* Row groups */
    metadata->num_row_groups = writer->num_row_groups;
    metadata->row_groups = carquet_arena_calloc(&writer->arena, writer->num_row_groups,
        sizeof(parquet_row_group_t));

    if (!metadata->row_groups && writer->num_row_groups > 0) {
        return CARQUET_ERROR_OUT_OF_MEMORY;
    }

    for (int32_t i = 0; i < writer->num_row_groups; i++) {
        const row_group_info_t* src_rg = &writer->row_groups[i];
        parquet_row_group_t* dst_rg = &metadata->row_groups[i];

        dst_rg->num_rows = src_rg->num_rows;
        dst_rg->total_byte_size = src_rg->total_byte_size;
        dst_rg->has_file_offset = true;
        dst_rg->file_offset = src_rg->file_offset;
        dst_rg->has_total_compressed_size = true;
        dst_rg->total_compressed_size = src_rg->total_compressed_size;
        dst_rg->has_ordinal = true;
        dst_rg->ordinal = src_rg->ordinal;
        dst_rg->num_columns = src_rg->num_columns;
        dst_rg->columns = carquet_arena_calloc(&writer->arena, src_rg->num_columns,
            sizeof(parquet_column_chunk_t));
        if (!dst_rg->columns && src_rg->num_columns > 0) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }

        for (int32_t j = 0; j < src_rg->num_columns; j++) {
            const row_group_column_info_t* src_col = &src_rg->columns[j];
            parquet_column_chunk_t* dst_chunk = &dst_rg->columns[j];
            parquet_column_metadata_t* meta = &dst_chunk->metadata;

            dst_chunk->file_offset = src_col->file_offset;
            dst_chunk->has_metadata = true;

            meta->type = src_col->type;
            meta->codec = src_col->codec;
            meta->num_values = src_col->num_values;
            meta->total_compressed_size = src_col->total_compressed_size;
            meta->total_uncompressed_size = src_col->total_uncompressed_size;
            meta->data_page_offset = src_col->data_page_offset;
            meta->num_encodings = 2;
            meta->encodings = writer->column_encodings[j];
            meta->path_len = writer->column_path_lens[j];
            meta->path_in_schema = writer->column_paths[j];

            meta->has_bloom_filter_offset = src_col->has_bloom_filter_offset;
            meta->bloom_filter_offset = src_col->bloom_filter_offset;
            meta->has_bloom_filter_length = src_col->has_bloom_filter_length;
            meta->bloom_filter_length = src_col->bloom_filter_length;

            dst_chunk->has_column_index_offset = src_col->has_column_index_offset;
            dst_chunk->column_index_offset = src_col->column_index_offset;
            dst_chunk->has_column_index_length = src_col->has_column_index_length;
            dst_chunk->column_index_length = src_col->column_index_length;
            dst_chunk->has_offset_index_offset = src_col->has_offset_index_offset;
            dst_chunk->offset_index_offset = src_col->offset_index_offset;
            dst_chunk->has_offset_index_length = src_col->has_offset_index_length;
            dst_chunk->offset_index_length = src_col->offset_index_length;
        }
    }

    return CARQUET_OK;
}

/* ============================================================================
 * Public API Implementation
 * ============================================================================
 */

carquet_writer_t* carquet_writer_create(
    const char* path,
    const carquet_schema_t* schema,
    const carquet_writer_options_t* options,
    carquet_error_t* error) {

    carquet_writer_t* writer = calloc(1, sizeof(carquet_writer_t));
    if (!writer) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate writer");
        return NULL;
    }

    /* Initialize arena */
    if (carquet_arena_init_size(&writer->arena, 4096) != CARQUET_OK) {
        free(writer);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate arena");
        return NULL;
    }

    /* Open file */
    writer->file = fopen(path, "wb");
    if (!writer->file) {
        carquet_arena_destroy(&writer->arena);
        free(writer);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_FILE_OPEN, "Failed to open file for writing: %s", path);
        return NULL;
    }
    writer->owns_file = true;

    writer->path = strdup(path);
    if (!writer->path) {
        fclose(writer->file);
        carquet_arena_destroy(&writer->arena);
        free(writer);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate path");
        return NULL;
    }

    /* Copy options */
    if (options) {
        writer->options = *options;
    } else {
        carquet_writer_options_init(&writer->options);
    }

    /* Store full schema elements for metadata serialization */
    {
        carquet_status_t status = store_schema_elements(writer, schema);
        if (status != CARQUET_OK) {
            carquet_writer_abort(writer);
            CARQUET_SET_ERROR(error, status, "Failed to store schema elements");
            return NULL;
        }

        status = build_column_metadata_cache(writer, schema);
        if (status != CARQUET_OK) {
            carquet_writer_abort(writer);
            CARQUET_SET_ERROR(error, status, "Failed to build writer metadata cache");
            return NULL;
        }
    }

    /* Add leaf columns from schema (schema is nonnull per API contract) */
    for (int32_t i = 0; i < schema->num_leaves; i++) {
        int32_t elem_idx = schema->leaf_indices[i];
        parquet_schema_element_t* elem = &schema->elements[elem_idx];

        carquet_logical_type_t* lt = elem->has_logical_type ? &elem->logical_type : NULL;

        carquet_status_t status = add_column_internal(
            writer,
            elem->name,
            elem->type,
            lt,
            elem->repetition_type,
            elem->type_length,
            schema->max_def_levels[i],
            schema->max_rep_levels[i]);

        if (status != CARQUET_OK) {
            carquet_writer_abort(writer);
            CARQUET_SET_ERROR(error, status, "Failed to add column from schema");
            return NULL;
        }
    }

    return writer;
}

carquet_writer_t* carquet_writer_create_file(
    FILE* file,
    const carquet_schema_t* schema,
    const carquet_writer_options_t* options,
    carquet_error_t* error) {

    /* file and schema are nonnull per API contract */
    carquet_writer_t* writer = calloc(1, sizeof(carquet_writer_t));
    if (!writer) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate writer");
        return NULL;
    }

    /* Initialize arena */
    if (carquet_arena_init_size(&writer->arena, 4096) != CARQUET_OK) {
        free(writer);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate arena");
        return NULL;
    }

    writer->file = file;
    writer->owns_file = false;

    /* Copy options */
    if (options) {
        writer->options = *options;
    } else {
        carquet_writer_options_init(&writer->options);
    }

    /* Store full schema elements for metadata serialization */
    {
        carquet_status_t status = store_schema_elements(writer, schema);
        if (status != CARQUET_OK) {
            carquet_writer_abort(writer);
            CARQUET_SET_ERROR(error, status, "Failed to store schema elements");
            return NULL;
        }

        status = build_column_metadata_cache(writer, schema);
        if (status != CARQUET_OK) {
            carquet_writer_abort(writer);
            CARQUET_SET_ERROR(error, status, "Failed to build writer metadata cache");
            return NULL;
        }
    }

    /* Add leaf columns from schema (schema is nonnull per API contract) */
    for (int32_t i = 0; i < schema->num_leaves; i++) {
        int32_t elem_idx = schema->leaf_indices[i];
        parquet_schema_element_t* elem = &schema->elements[elem_idx];

        carquet_logical_type_t* lt = elem->has_logical_type ? &elem->logical_type : NULL;

        carquet_status_t status = add_column_internal(
            writer,
            elem->name,
            elem->type,
            lt,
            elem->repetition_type,
            elem->type_length,
            schema->max_def_levels[i],
            schema->max_rep_levels[i]);

        if (status != CARQUET_OK) {
            carquet_writer_abort(writer);
            CARQUET_SET_ERROR(error, status, "Failed to add column from schema");
            return NULL;
        }
    }

    return writer;
}

carquet_status_t carquet_writer_write_batch(
    carquet_writer_t* writer,
    int32_t column_index,
    const void* values,
    int64_t num_values,
    const int16_t* def_levels,
    const int16_t* rep_levels) {

    /* writer and values are nonnull per API contract */
    if (column_index < 0 || column_index >= writer->num_columns) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    /* Ensure header is written */
    carquet_status_t status = ensure_header_written(writer);
    if (status != CARQUET_OK) {
        return status;
    }

    /* Ensure we have a row group */
    status = ensure_row_group(writer);
    if (status != CARQUET_OK) {
        return status;
    }

    /* Write to the row group */
    status = carquet_row_group_writer_write_column(
        writer->current_row_group,
        column_index,
        values,
        num_values,
        def_levels,
        rep_levels);

    if (status != CARQUET_OK) {
        return status;
    }

    writer->column_values_written[column_index] += num_values;

    /* Track rows (use column 0 as reference).
     * For repeated columns (max_rep_level > 0), the number of logical rows
     * is the count of rep_level == 0 entries (new top-level records).
     * For non-repeated columns, num_values == num_rows. */
    if (column_index == 0) {
        if (rep_levels && writer->columns[0].max_rep_level > 0) {
            int64_t rows = 0;
            for (int64_t i = 0; i < num_values; i++) {
                if (rep_levels[i] == 0) rows++;
            }
            writer->current_row_group_rows += rows;
        } else {
            writer->current_row_group_rows += num_values;
        }
    }

    return CARQUET_OK;
}

carquet_status_t carquet_writer_new_row_group(carquet_writer_t* writer) {
    /* writer is nonnull per API contract */
    /* Ensure header is written */
    carquet_status_t status = ensure_header_written(writer);
    if (status != CARQUET_OK) {
        return status;
    }

    /* Flush current row group if any */
    return flush_row_group(writer);
}

carquet_status_t carquet_writer_close(carquet_writer_t* writer) {
    /* writer is nonnull per API contract */
    carquet_status_t status = CARQUET_OK;

    /* Ensure header is written */
    status = ensure_header_written(writer);
    if (status != CARQUET_OK) {
        goto cleanup;
    }

    /* Flush any pending row group */
    status = flush_row_group(writer);
    if (status != CARQUET_OK) {
        goto cleanup;
    }

    /* Build file metadata */
    parquet_file_metadata_t metadata;
    status = build_file_metadata(writer, &metadata);
    if (status != CARQUET_OK) {
        goto cleanup;
    }

    /* Serialize metadata to buffer */
    carquet_buffer_t metadata_buffer;
    carquet_buffer_init(&metadata_buffer);

    status = parquet_write_file_metadata(&metadata, &metadata_buffer, NULL);
    if (status != CARQUET_OK) {
        carquet_buffer_destroy(&metadata_buffer);
        goto cleanup;
    }

    /* Write metadata */
    if (fwrite(metadata_buffer.data, 1, metadata_buffer.size, writer->file) != metadata_buffer.size) {
        carquet_buffer_destroy(&metadata_buffer);
        status = CARQUET_ERROR_FILE_WRITE;
        goto cleanup;
    }

    /* Write metadata length (4 bytes, little-endian) */
    uint32_t metadata_len = (uint32_t)metadata_buffer.size;
    uint8_t len_bytes[4];
    len_bytes[0] = (uint8_t)(metadata_len & 0xFF);
    len_bytes[1] = (uint8_t)((metadata_len >> 8) & 0xFF);
    len_bytes[2] = (uint8_t)((metadata_len >> 16) & 0xFF);
    len_bytes[3] = (uint8_t)((metadata_len >> 24) & 0xFF);

    if (fwrite(len_bytes, 1, 4, writer->file) != 4) {
        carquet_buffer_destroy(&metadata_buffer);
        status = CARQUET_ERROR_FILE_WRITE;
        goto cleanup;
    }

    carquet_buffer_destroy(&metadata_buffer);

    /* Write footer magic */
    status = write_magic(writer->file);
    if (status != CARQUET_OK) {
        goto cleanup;
    }

cleanup:
    /* Free resources */
    if (writer->current_row_group) {
        carquet_row_group_writer_destroy(writer->current_row_group);
        writer->current_row_group = NULL;
    }

    if (writer->owns_file && writer->file) {
        fclose(writer->file);
        writer->file = NULL;
    }

    /* Free column definitions */
    if (writer->columns) {
        for (int32_t i = 0; i < writer->num_columns; i++) {
            free(writer->columns[i].name);
        }
        free(writer->columns);
    }

    /* Free schema elements */
    if (writer->schema_elements) {
        for (int32_t i = 0; i < writer->num_schema_elements; i++) {
            free(writer->schema_elements[i].name);
        }
        free(writer->schema_elements);
    }
    if (writer->column_paths) {
        for (int32_t i = 0; i < writer->num_columns; i++) {
            free(writer->column_paths[i]);
        }
        free(writer->column_paths);
    }
    free(writer->column_path_lens);
    free(writer->column_encodings);

    free(writer->column_values_written);
    if (writer->row_groups) {
        for (int32_t i = 0; i < writer->num_row_groups; i++) {
            free(writer->row_groups[i].columns);
        }
    }
    free(writer->row_groups);
    free(writer->path);
    carquet_arena_destroy(&writer->arena);
    free(writer);

    return status;
}

void carquet_writer_abort(carquet_writer_t* writer) {
    if (!writer) return;

    /* Cleanup row group */
    if (writer->current_row_group) {
        carquet_row_group_writer_destroy(writer->current_row_group);
        writer->current_row_group = NULL;
    }

    /* Close and delete file */
    if (writer->owns_file && writer->file) {
        fclose(writer->file);
        writer->file = NULL;

        if (writer->path) {
            remove(writer->path);
        }
    }

    /* Free column definitions */
    if (writer->columns) {
        for (int32_t i = 0; i < writer->num_columns; i++) {
            free(writer->columns[i].name);
        }
        free(writer->columns);
    }

    /* Free schema elements */
    if (writer->schema_elements) {
        for (int32_t i = 0; i < writer->num_schema_elements; i++) {
            free(writer->schema_elements[i].name);
        }
        free(writer->schema_elements);
    }
    if (writer->column_paths) {
        for (int32_t i = 0; i < writer->num_columns; i++) {
            free(writer->column_paths[i]);
        }
        free(writer->column_paths);
    }
    free(writer->column_path_lens);
    free(writer->column_encodings);

    free(writer->column_values_written);
    if (writer->row_groups) {
        for (int32_t i = 0; i < writer->num_row_groups; i++) {
            free(writer->row_groups[i].columns);
        }
    }
    free(writer->row_groups);
    free(writer->path);
    carquet_arena_destroy(&writer->arena);
    free(writer);
}
