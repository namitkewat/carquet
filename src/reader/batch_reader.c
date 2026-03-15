/**
 * @file batch_reader.c
 * @brief High-level batch reader with column projection and parallel I/O
 *
 * This provides a production-ready API for efficiently reading Parquet files
 * with support for:
 * - Column projection (only read needed columns)
 * - Parallel column reading
 * - Memory-mapped I/O
 * - Batched output
 * - Buffer pooling to minimize allocations
 */

#include <carquet/carquet.h>
#include "reader_internal.h"
#include "worker_pool.h"
#include "core/arena.h"
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* SIMD dispatch function for null bitmap construction */
extern void carquet_dispatch_build_null_bitmap(const int16_t* def_levels, int64_t count,
                                                int16_t max_def_level, uint8_t* null_bitmap);

/* ============================================================================
 * Internal Structures
 * ============================================================================
 */

typedef struct carquet_column_data {
    void* data;                 /* Column values (or uint32_t* indices if dict preserved) */
    uint8_t* null_bitmap;       /* Null bitmap (1 bit per value), NULL for REQUIRED */
    int64_t num_values;         /* Number of values */
    size_t data_capacity;       /* Allocated capacity for data */
    carquet_physical_type_t type;
    int32_t type_length;        /* For fixed-length types */
    carquet_data_ownership_t ownership;  /* OWNED or VIEW (for future zero-copy) */

    /* Dictionary preservation (when config.preserve_dictionaries == true) */
    bool is_dictionary;             /* True if this column has preserved dictionary */
    const uint8_t* dictionary_data; /* Pointer to dictionary bytes (view, not owned) */
    int32_t dictionary_count;       /* Number of dictionary entries */
    const uint32_t* dictionary_offsets; /* Offset table for BYTE_ARRAY (view) */
} carquet_column_data_t;

/* Pre-allocated column buffer pool for reuse across batches */
typedef struct carquet_column_pool {
    void* data;                 /* Pre-allocated data buffer */
    size_t data_capacity;       /* Capacity in bytes */
    uint8_t* null_bitmap;       /* Pre-allocated null bitmap */
    size_t bitmap_capacity;     /* Capacity in bytes */
    int16_t* def_levels;        /* Pre-allocated def levels buffer */
    size_t def_levels_capacity; /* Capacity in elements */
} carquet_column_pool_t;

struct carquet_row_batch {
    carquet_column_data_t* columns;
    int32_t num_columns;
    int64_t num_rows;
    carquet_arena_t arena;
    bool pooled;                /* If true, data buffers are from batch_reader pool */
};

/* Pipeline ring buffer slot: holds pre-read column data for one RG */
typedef struct {
    carquet_column_reader_t** col_readers;  /* [num_projected] readers, used for bulk read */
    int32_t rg_index;                       /* row group index, -1 = empty */
    bool ready;                             /* all columns fully read */

    /* Pre-read value buffers (entire column chunk per column) */
    void** col_values;       /* [num_projected] value buffers */
    size_t* col_buf_sizes;   /* [num_projected] buffer capacities in bytes */
    int64_t* col_num_values; /* [num_projected] values actually read */
    int64_t total_rows;      /* total rows in this RG (min of col values read) */
    int64_t rows_consumed;   /* rows already served to batch_reader_next */
} rg_slot_t;

/* Task argument for parallel bulk column reading */
typedef struct {
    carquet_column_reader_t* col_reader;
    void* dest;
    int64_t max_values;
    int64_t* out_values_read;
} bulk_read_arg_t;

struct carquet_batch_reader {
    carquet_reader_t* reader;
    carquet_batch_reader_config_t config;

    /* Column projection */
    int32_t* projected_columns;  /* File column indices to read */
    int32_t num_projected;       /* Number of projected columns */
    carquet_physical_type_t* projected_types;
    int32_t* projected_type_lengths;
    int16_t* projected_max_defs;
    size_t* projected_value_sizes;

    /* Reading state */
    int32_t current_row_group;
    int64_t rows_read_in_group;
    int64_t total_rows_read;

    /* Column readers for current row group */
    carquet_column_reader_t** col_readers;

    /* Memory-mapped data */
    uint8_t* mmap_data;
    size_t mmap_size;

    /* Buffer pool for reuse across batches (one per projected column) */
    carquet_column_pool_t* col_pools;

    /* Cached batch struct to avoid repeated alloc/free */
    carquet_row_batch_t* cached_batch;

    /* Persistent worker pool for cross-RG parallel decompression */
    carquet_worker_pool_t* pool;

    /* Pipeline ring buffer for multi-RG parallel decompression.
     * Pre-decompresses pages for upcoming row groups so that by the time
     * batch_reader_next() needs data, it's already decompressed. */
    rg_slot_t* pipeline;          /* [pipeline_depth] ring buffer */
    int32_t pipeline_depth;       /* window size */
    int32_t pipeline_head;        /* next slot to consume */
    int32_t pipeline_count;       /* slots in use */
    int32_t* rg_order;            /* pre-filtered list of RG indices */
    int32_t rg_order_len;         /* total filtered RGs */
    int32_t rg_order_next;        /* next RG to submit */
    bool pipeline_active;         /* multi-RG pipeline enabled */
};

/* ============================================================================
 * Configuration
 * ============================================================================
 */

void carquet_batch_reader_config_init(carquet_batch_reader_config_t* config) {
    /* config is nonnull per API contract */
    memset(config, 0, sizeof(*config));
    config->batch_size = 65536;  /* 64K rows per batch */
    config->num_threads = 0;     /* Auto-detect */
    config->use_mmap = false;
}

/* ============================================================================
 * Helper Functions
 * ============================================================================
 */

/* Maximum reasonable type_length for FIXED_LEN_BYTE_ARRAY (16 MB) */
#define CARQUET_MAX_TYPE_LENGTH (16 * 1024 * 1024)

static size_t get_type_size(carquet_physical_type_t type, int32_t type_length) {
    switch (type) {
        case CARQUET_PHYSICAL_BOOLEAN: return 1;
        case CARQUET_PHYSICAL_INT32: return 4;
        case CARQUET_PHYSICAL_INT64: return 8;
        case CARQUET_PHYSICAL_INT96: return 12;
        case CARQUET_PHYSICAL_FLOAT: return 4;
        case CARQUET_PHYSICAL_DOUBLE: return 8;
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
            /* Validate type_length to prevent overflow attacks */
            if (type_length <= 0 || type_length > CARQUET_MAX_TYPE_LENGTH) {
                return 0;  /* Invalid - will cause allocation to fail safely */
            }
            return (size_t)type_length;
        case CARQUET_PHYSICAL_BYTE_ARRAY: return sizeof(carquet_byte_array_t);
        default: return 0;
    }
}

static int resolve_column_name(const carquet_reader_t* reader, const char* name) {
    const carquet_schema_t* schema = carquet_reader_schema(reader);
    if (!schema) return -1;

    return carquet_schema_find_column(schema, name);
}

/* Ensure a pool buffer is at least 'needed' bytes, growing if necessary */
static void* pool_ensure_data(carquet_column_pool_t* pool, size_t needed) {
    if (needed <= pool->data_capacity) {
        return pool->data;
    }
    free(pool->data);
    pool->data = malloc(needed);
    pool->data_capacity = pool->data ? needed : 0;
    return pool->data;
}

static uint8_t* pool_ensure_bitmap(carquet_column_pool_t* pool, size_t needed) {
    if (needed <= pool->bitmap_capacity) {
        memset(pool->null_bitmap, 0, needed);
        return pool->null_bitmap;
    }
    free(pool->null_bitmap);
    pool->null_bitmap = calloc(1, needed);
    pool->bitmap_capacity = pool->null_bitmap ? needed : 0;
    return pool->null_bitmap;
}

static int16_t* pool_ensure_def_levels(carquet_column_pool_t* pool, size_t count) {
    if (count <= pool->def_levels_capacity) {
        return pool->def_levels;
    }
    free(pool->def_levels);
    pool->def_levels = malloc(sizeof(int16_t) * count);
    pool->def_levels_capacity = pool->def_levels ? count : 0;
    return pool->def_levels;
}

static bool column_can_zero_copy_batch(
    const carquet_column_reader_t* col_reader,
    carquet_physical_type_t type,
    int16_t max_def,
    int64_t rows_to_read) {

    if (!col_reader || !col_reader->page_loaded ||
        col_reader->decoded_ownership != CARQUET_DATA_VIEW ||
        max_def != 0 || col_reader->max_rep_level > 0 ||
        type == CARQUET_PHYSICAL_BYTE_ARRAY) {
        return false;
    }

    int32_t page_available = col_reader->page_num_values - col_reader->page_values_read;
    return page_available > 0 && page_available >= (int32_t)rows_to_read;
}

static int64_t column_zero_copy_rows_available(
    const carquet_column_reader_t* col_reader,
    carquet_physical_type_t type,
    int16_t max_def) {

    if (!col_reader || !col_reader->page_loaded ||
        col_reader->decoded_ownership != CARQUET_DATA_VIEW ||
        max_def != 0 || col_reader->max_rep_level > 0 ||
        type == CARQUET_PHYSICAL_BYTE_ARRAY) {
        return 0;
    }

    int32_t page_available = col_reader->page_num_values - col_reader->page_values_read;
    return page_available > 0 ? page_available : 0;
}

static int64_t clamp_rows_to_zero_copy_window(
    const carquet_batch_reader_t* batch_reader,
    int64_t rows_to_read) {

    int64_t zero_copy_rows = rows_to_read;

    for (int32_t i = 0; i < batch_reader->num_projected; i++) {
        int64_t page_rows = column_zero_copy_rows_available(
            batch_reader->col_readers[i],
            batch_reader->projected_types[i],
            batch_reader->projected_max_defs[i]);

        if (page_rows <= 0) {
            return rows_to_read;
        }
        if (page_rows < zero_copy_rows) {
            zero_copy_rows = page_rows;
        }
    }

    return zero_copy_rows;
}

static bool column_is_zero_copy_candidate(
    const carquet_column_reader_t* col_reader,
    carquet_physical_type_t type,
    int16_t max_def) {

    if (!col_reader || !col_reader->file_reader ||
        col_reader->file_reader->mmap_data == NULL ||
        !col_reader->col_meta ||
        max_def != 0 || col_reader->max_rep_level > 0 ||
        col_reader->col_meta->codec != CARQUET_COMPRESSION_UNCOMPRESSED) {
        return false;
    }

    switch (type) {
        case CARQUET_PHYSICAL_INT32:
        case CARQUET_PHYSICAL_INT64:
        case CARQUET_PHYSICAL_INT96:
        case CARQUET_PHYSICAL_FLOAT:
        case CARQUET_PHYSICAL_DOUBLE:
        case CARQUET_PHYSICAL_FIXED_LEN_BYTE_ARRAY:
            return true;
        case CARQUET_PHYSICAL_BOOLEAN:
        case CARQUET_PHYSICAL_BYTE_ARRAY:
        default:
            return false;
    }
}

static void read_projected_column(
    carquet_batch_reader_t* batch_reader,
    carquet_row_batch_t* new_batch,
    int32_t col_i,
    int64_t rows_to_read,
    bool* read_error) {

    if (*read_error) {
        return;
    }

    carquet_column_reader_t* col_reader = batch_reader->col_readers[col_i];
    carquet_column_data_t* col_data = &new_batch->columns[col_i];
    carquet_column_pool_t* pool = &batch_reader->col_pools[col_i];
    size_t value_size = batch_reader->projected_value_sizes[col_i];
    int16_t max_def = batch_reader->projected_max_defs[col_i];

    col_data->type = batch_reader->projected_types[col_i];
    col_data->type_length = batch_reader->projected_type_lengths[col_i];
    col_data->is_dictionary = false;
    col_data->dictionary_data = NULL;
    col_data->dictionary_count = 0;
    col_data->dictionary_offsets = NULL;

    /* Dictionary preservation: if enabled and column has a dictionary,
     * return uint32_t indices instead of materializing values */
    bool use_dict_preserve = batch_reader->config.preserve_dictionaries &&
                             col_reader->has_dictionary;

    /* Set the preserve_dictionary flag on the column reader so the page
     * decoder knows to skip materialization */
    col_reader->preserve_dictionary = use_dict_preserve;

    /* When preserving dictionaries, value_size is sizeof(uint32_t) for indices */
    size_t effective_value_size = use_dict_preserve ? sizeof(uint32_t) : value_size;

    /* Check if direct page handoff is possible:
     * - Column is REQUIRED (no nulls, no definition levels)
     * - Page loader can expose a stable view (mmap or reusable page buffer)
     * - Entire page slice fits in this batch
     * - Not in dictionary-preserve mode (indices layout differs)
     */
    bool try_zero_copy = (max_def == 0) &&
                         (!col_reader->page_loaded) &&
                         !use_dict_preserve;

    if (try_zero_copy) {
        /* Trigger page load to check if it's a zero-copy page */
        int64_t dummy_read = carquet_column_read_batch(
            col_reader, NULL, 0, NULL, NULL);
        (void)dummy_read;
    }

    bool use_zero_copy = !use_dict_preserve &&
                         column_can_zero_copy_batch(
                             col_reader, col_data->type, max_def, rows_to_read);

    if (use_zero_copy) {
        /* ====== ZERO-COPY PATH ====== */
        /* Point directly to the currently loaded page slice. */
        size_t byte_offset = (size_t)col_reader->page_values_read * value_size;
        col_data->data = (uint8_t*)col_reader->decoded_values + byte_offset;
        col_data->data_capacity = 0;  /* Not our allocation */
        col_data->ownership = CARQUET_DATA_VIEW;
        col_data->num_values = rows_to_read;

        /* No nulls in REQUIRED columns - return NULL bitmap */
        col_data->null_bitmap = NULL;

        /* Mark page as consumed */
        col_reader->page_values_read += (int32_t)rows_to_read;
        col_reader->values_remaining -= rows_to_read;
        return;
    }

    /* ====== STANDARD PATH (with copy, using pooled buffers) ====== */

    /* Validate value_size and check for overflow */
    if (effective_value_size == 0 || rows_to_read <= 0) {
        *read_error = true;
        return;
    }

    /* Check for multiplication overflow (max 1GB allocation) */
    #define CARQUET_MAX_BATCH_ALLOC (1024ULL * 1024 * 1024)
    if (effective_value_size > CARQUET_MAX_BATCH_ALLOC / (size_t)rows_to_read) {
        *read_error = true;
        return;
    }

    size_t data_size = effective_value_size * (size_t)rows_to_read;

    /* Use pooled data buffer (grows as needed, never shrinks) */
    col_data->data = pool_ensure_data(pool, data_size);
    if (!col_data->data) {
        *read_error = true;
        return;
    }
    col_data->data_capacity = data_size;
    col_data->ownership = CARQUET_DATA_VIEW;  /* Pool owns the buffer */

    /* Only allocate null bitmap for OPTIONAL columns */
    if (max_def > 0) {
        size_t bitmap_size = ((size_t)rows_to_read + 7) / 8;
        col_data->null_bitmap = pool_ensure_bitmap(pool, bitmap_size);
    } else {
        col_data->null_bitmap = NULL;  /* REQUIRED columns have no nulls */
    }

    /* Read values (reuse pooled def_levels buffer) */
    int16_t* def_levels = NULL;
    if (max_def > 0) {
        def_levels = pool_ensure_def_levels(pool, (size_t)rows_to_read);
    }

    int64_t values_read = carquet_column_read_batch(
        col_reader, col_data->data, rows_to_read, def_levels, NULL);

    if (values_read < 0) {
        *read_error = true;
        return;
    }

    col_data->num_values = values_read;

    /* Attach dictionary metadata for preserved dictionary columns */
    if (use_dict_preserve) {
        col_data->is_dictionary = true;
        col_data->dictionary_data = col_reader->dictionary_data;
        col_data->dictionary_count = col_reader->dictionary_count;
        col_data->dictionary_offsets = col_reader->dictionary_offsets;
    }

    /* The column reader returns dense non-null values (Parquet convention).
     * The batch reader's contract is row-aligned: value[i] corresponds to
     * logical row i, with null slots zeroed.  Expand in-place (back-to-front)
     * so the data and null bitmap are consistent. */
    if (def_levels && max_def > 0 && values_read > 0) {
        int64_t non_null = 0;
        for (int64_t k = 0; k < values_read; k++) {
            if (def_levels[k] == max_def) non_null++;
        }

        if (non_null < values_read) {
            uint8_t* data = (uint8_t*)col_data->data;
            int64_t src = non_null - 1;
            for (int64_t dst = values_read - 1; dst >= 0; dst--) {
                uint8_t* dp = data + (size_t)dst * effective_value_size;
                if (def_levels[dst] == max_def) {
                    uint8_t* sp = data + (size_t)src * effective_value_size;
                    if (sp != dp) memmove(dp, sp, effective_value_size);
                    src--;
                } else {
                    memset(dp, 0, effective_value_size);
                }
            }
        }
    }

    /* Build null bitmap from definition levels (uses SIMD when available) */
    if (def_levels && col_data->null_bitmap) {
        carquet_dispatch_build_null_bitmap(def_levels, values_read,
                                           max_def, col_data->null_bitmap);
    }
}

/* ============================================================================
 * Column Reader Reset (reuse across row groups)
 * ============================================================================
 */

static void reset_column_reader_for_row_group(
    carquet_column_reader_t* col_reader,
    carquet_reader_t* file_reader,
    int32_t row_group_index,
    int32_t column_index) {

    const parquet_row_group_t* rg = &file_reader->metadata.row_groups[row_group_index];

    col_reader->row_group_index = row_group_index;
    col_reader->column_index = column_index;

    if (!rg->columns || column_index >= rg->num_columns) {
        col_reader->chunk = NULL;
        col_reader->col_meta = NULL;
        col_reader->values_remaining = 0;
        return;
    }

    col_reader->chunk = &rg->columns[column_index];
    if (!col_reader->chunk->has_metadata) {
        col_reader->col_meta = NULL;
        col_reader->values_remaining = 0;
        return;
    }
    col_reader->col_meta = &col_reader->chunk->metadata;

    /* Reset reading state */
    col_reader->values_remaining = col_reader->col_meta->num_values;
    col_reader->data_start_offset = col_reader->col_meta->data_page_offset;
    col_reader->current_page = 0;
    col_reader->page_loaded = false;
    col_reader->page_num_values = 0;
    col_reader->page_values_read = 0;
    col_reader->page_header_size = 0;
    col_reader->page_compressed_size = 0;

    /* Dictionary may differ between row groups - must reload */
    if (col_reader->has_dictionary) {
        if (col_reader->dictionary_ownership == CARQUET_DATA_OWNED) {
            free(col_reader->dictionary_data);
        }
        free(col_reader->dictionary_offsets);
        col_reader->dictionary_data = NULL;
        col_reader->dictionary_offsets = NULL;
        col_reader->dictionary_size = 0;
        col_reader->dictionary_count = 0;
        col_reader->dictionary_ownership = CARQUET_DATA_OWNED;
        col_reader->has_dictionary = false;
    }

    /* If decoded_values is a VIEW (mmap pointer), don't free - just clear */
    if (col_reader->decoded_ownership == CARQUET_DATA_VIEW) {
        col_reader->decoded_values = NULL;
        col_reader->decoded_capacity = 0;
    }
    col_reader->decoded_ownership = CARQUET_DATA_OWNED;

    /* Free BYTE_ARRAY page data retention buffer */
    free(col_reader->page_data_for_values);
    col_reader->page_data_for_values = NULL;

    /* Keep reusable buffers: decoded_values, decoded_def_levels,
     * decoded_rep_levels, indices_buffer, decompress_buffer.
     * These will be reused on the next page load. */
}

/* ============================================================================
 * Batch Reader Implementation
 * ============================================================================
 */

carquet_batch_reader_t* carquet_batch_reader_create(
    carquet_reader_t* reader,
    const carquet_batch_reader_config_t* config,
    carquet_error_t* error) {

    /* reader is nonnull per API contract */
    carquet_batch_reader_t* batch_reader = calloc(1, sizeof(carquet_batch_reader_t));
    if (!batch_reader) {
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate batch reader");
        return NULL;
    }

    batch_reader->reader = reader;

    /* Copy config or use defaults */
    if (config) {
        batch_reader->config = *config;
    } else {
        carquet_batch_reader_config_init(&batch_reader->config);
    }

    /* Resolve column projection */
    int32_t total_columns = carquet_reader_num_columns(reader);

    if (batch_reader->config.column_indices && batch_reader->config.num_columns > 0) {
        /* Use provided column indices */
        batch_reader->num_projected = batch_reader->config.num_columns;
        batch_reader->projected_columns = malloc(sizeof(int32_t) * batch_reader->num_projected);
        if (!batch_reader->projected_columns) {
            free(batch_reader);
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate projection");
            return NULL;
        }
        memcpy(batch_reader->projected_columns, batch_reader->config.column_indices,
               sizeof(int32_t) * batch_reader->num_projected);
    } else if (batch_reader->config.column_names && batch_reader->config.num_column_names > 0) {
        /* Resolve column names to indices */
        batch_reader->num_projected = batch_reader->config.num_column_names;
        batch_reader->projected_columns = malloc(sizeof(int32_t) * batch_reader->num_projected);
        if (!batch_reader->projected_columns) {
            free(batch_reader);
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate projection");
            return NULL;
        }

        for (int32_t i = 0; i < batch_reader->num_projected; i++) {
            const char* col_name = batch_reader->config.column_names[i];
            int32_t idx = resolve_column_name(reader, col_name);
            if (idx < 0) {
                free(batch_reader->projected_columns);
                free(batch_reader);
                CARQUET_SET_ERROR(error, CARQUET_ERROR_COLUMN_NOT_FOUND,
                    "Column not found: %s", col_name);
                return NULL;
            }
            batch_reader->projected_columns[i] = idx;
        }
    } else {
        /* Read all columns */
        batch_reader->num_projected = total_columns;
        batch_reader->projected_columns = malloc(sizeof(int32_t) * total_columns);
        if (!batch_reader->projected_columns) {
            free(batch_reader);
            CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate projection");
            return NULL;
        }
        for (int32_t i = 0; i < total_columns; i++) {
            batch_reader->projected_columns[i] = i;
        }
    }

    /* Allocate column reader array */
    batch_reader->col_readers = calloc(batch_reader->num_projected,
                                        sizeof(carquet_column_reader_t*));
    if (!batch_reader->col_readers) {
        free(batch_reader->projected_columns);
        free(batch_reader);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate column readers");
        return NULL;
    }

    batch_reader->projected_types = malloc(sizeof(carquet_physical_type_t) *
                                           (size_t)batch_reader->num_projected);
    batch_reader->projected_type_lengths = malloc(sizeof(int32_t) *
                                                  (size_t)batch_reader->num_projected);
    batch_reader->projected_max_defs = malloc(sizeof(int16_t) *
                                              (size_t)batch_reader->num_projected);
    batch_reader->projected_value_sizes = malloc(sizeof(size_t) *
                                                 (size_t)batch_reader->num_projected);
    if (!batch_reader->projected_types || !batch_reader->projected_type_lengths ||
        !batch_reader->projected_max_defs || !batch_reader->projected_value_sizes) {
        free(batch_reader->projected_value_sizes);
        free(batch_reader->projected_max_defs);
        free(batch_reader->projected_type_lengths);
        free(batch_reader->projected_types);
        free(batch_reader->col_readers);
        free(batch_reader->projected_columns);
        free(batch_reader);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate projection metadata");
        return NULL;
    }

    {
        const carquet_schema_t* schema = carquet_reader_schema(reader);
        for (int32_t i = 0; i < batch_reader->num_projected; i++) {
            int32_t file_col_idx = batch_reader->projected_columns[i];
            int32_t schema_idx = schema->leaf_indices[file_col_idx];
            const parquet_schema_element_t* elem = &schema->elements[schema_idx];

            batch_reader->projected_types[i] =
                elem->has_type ? elem->type : CARQUET_PHYSICAL_BYTE_ARRAY;
            batch_reader->projected_type_lengths[i] = elem->type_length;
            batch_reader->projected_max_defs[i] = schema->max_def_levels[file_col_idx];
            batch_reader->projected_value_sizes[i] = get_type_size(
                batch_reader->projected_types[i],
                batch_reader->projected_type_lengths[i]);
        }
    }

    /* Allocate buffer pool (one per projected column) */
    batch_reader->col_pools = calloc(batch_reader->num_projected,
                                      sizeof(carquet_column_pool_t));
    if (!batch_reader->col_pools) {
        free(batch_reader->projected_value_sizes);
        free(batch_reader->projected_max_defs);
        free(batch_reader->projected_type_lengths);
        free(batch_reader->projected_types);
        free(batch_reader->col_readers);
        free(batch_reader->projected_columns);
        free(batch_reader);
        CARQUET_SET_ERROR(error, CARQUET_ERROR_OUT_OF_MEMORY, "Failed to allocate buffer pool");
        return NULL;
    }

    batch_reader->current_row_group = -1;

    /* ====================================================================
     * Pre-compute filtered row group order
     * ==================================================================== */
    int32_t num_row_groups = carquet_reader_num_row_groups(reader);
    batch_reader->rg_order = malloc(sizeof(int32_t) * (size_t)num_row_groups);
    if (!batch_reader->rg_order) {
        batch_reader->rg_order_len = 0;
    } else {
        int32_t count = 0;
        for (int32_t rg = 0; rg < num_row_groups; rg++) {
            if (batch_reader->config.row_group_filter) {
                bool should_read = batch_reader->config.row_group_filter(
                    reader, rg, batch_reader->config.row_group_filter_ctx);
                if (!should_read) continue;
            }
            batch_reader->rg_order[count++] = rg;
        }
        batch_reader->rg_order_len = count;
    }
    batch_reader->rg_order_next = 0;

    /* ====================================================================
     * Create worker pool + pipeline for compressed mmap multi-RG files
     * ==================================================================== */
    if (reader->mmap_data != NULL && batch_reader->rg_order_len > 1) {
        bool has_compression = false;
        const parquet_file_metadata_t* meta = &reader->metadata;
        if (meta->num_row_groups > 0 && meta->row_groups[0].columns) {
            for (int32_t ci = 0; ci < batch_reader->num_projected; ci++) {
                int32_t file_col = batch_reader->projected_columns[ci];
                if (file_col < meta->row_groups[0].num_columns) {
                    const parquet_column_chunk_t* chunk = &meta->row_groups[0].columns[file_col];
                    if (chunk->has_metadata &&
                        chunk->metadata.codec != CARQUET_COMPRESSION_UNCOMPRESSED) {
                        has_compression = true;
                        break;
                    }
                }
            }
        }

        if (has_compression) {
            int32_t pt = batch_reader->config.num_threads;
#ifdef _OPENMP
            if (pt <= 0) pt = omp_get_max_threads();
#else
            if (pt <= 0) pt = 4;
#endif
            if (pt < 2) pt = 2;

            batch_reader->pool = carquet_worker_pool_create(pt);
            if (batch_reader->pool) {
                int32_t depth = batch_reader->rg_order_len;
                if (depth > pt * 2) depth = pt * 2;
                if (depth < 1) depth = 1;

                batch_reader->pipeline = calloc(depth, sizeof(rg_slot_t));
                if (batch_reader->pipeline) {
                    batch_reader->pipeline_depth = depth;
                    batch_reader->pipeline_head = 0;
                    batch_reader->pipeline_count = 0;
                    batch_reader->pipeline_active = true;

                    int32_t np = batch_reader->num_projected;
                    bool alloc_ok = true;
                    for (int32_t s = 0; s < depth && alloc_ok; s++) {
                        batch_reader->pipeline[s].col_readers = calloc(np, sizeof(carquet_column_reader_t*));
                        batch_reader->pipeline[s].col_values = calloc(np, sizeof(void*));
                        batch_reader->pipeline[s].col_buf_sizes = calloc(np, sizeof(size_t));
                        batch_reader->pipeline[s].col_num_values = calloc(np, sizeof(int64_t));
                        batch_reader->pipeline[s].rg_index = -1;
                        if (!batch_reader->pipeline[s].col_readers ||
                            !batch_reader->pipeline[s].col_values ||
                            !batch_reader->pipeline[s].col_buf_sizes ||
                            !batch_reader->pipeline[s].col_num_values) {
                            /* Cleanup all slots on failure */
                            for (int32_t j = 0; j <= s; j++) {
                                free(batch_reader->pipeline[j].col_readers);
                                free(batch_reader->pipeline[j].col_values);
                                free(batch_reader->pipeline[j].col_buf_sizes);
                                free(batch_reader->pipeline[j].col_num_values);
                            }
                            free(batch_reader->pipeline);
                            batch_reader->pipeline = NULL;
                            batch_reader->pipeline_active = false;
                            alloc_ok = false;
                        }
                    }
                }
            }
        }
    }

    return batch_reader;
}

static carquet_status_t open_row_group_readers(
    carquet_batch_reader_t* batch_reader,
    int32_t row_group_index,
    carquet_error_t* error) {

    /* Reuse existing readers if possible, otherwise create new ones */
    for (int32_t i = 0; i < batch_reader->num_projected; i++) {
        int32_t file_col_idx = batch_reader->projected_columns[i];

        if (batch_reader->col_readers[i]) {
            /* Reuse: reset state but keep allocated buffers */
            reset_column_reader_for_row_group(
                batch_reader->col_readers[i],
                batch_reader->reader,
                row_group_index, file_col_idx);
        } else {
            /* First time: create new reader */
            batch_reader->col_readers[i] = carquet_reader_get_column(
                batch_reader->reader, row_group_index, file_col_idx, error);

            if (!batch_reader->col_readers[i]) {
                /* Close already opened readers */
                for (int32_t j = 0; j < i; j++) {
                    carquet_column_reader_free(batch_reader->col_readers[j]);
                    batch_reader->col_readers[j] = NULL;
                }
                return error ? error->code : CARQUET_ERROR_COLUMN_NOT_FOUND;
            }
        }
    }

    batch_reader->current_row_group = row_group_index;
    batch_reader->rows_read_in_group = 0;

    return CARQUET_OK;
}

/* ============================================================================
 * Pipeline Ring Buffer
 * ============================================================================
 *
 * Pre-decompresses pages for multiple row groups in parallel using the worker
 * pool. On the first call to batch_reader_next(), decompression tasks are
 * submitted for up to pipeline_depth row groups. As the user consumes
 * batches and exhausts a row group, that slot is retired and the next
 * uncovered RG is submitted. With pipeline_depth >= total_RGs (the common
 * benchmark case), ALL decompression happens upfront in parallel.
 */

/**
 * Bulk-read task: reads ALL values from a column reader into a pre-allocated
 * buffer. This forces decompression of ALL pages in the column chunk.
 */
static void bulk_read_task(void* arg) {
    bulk_read_arg_t* t = (bulk_read_arg_t*)arg;
    if (t->col_reader && t->dest && t->max_values > 0) {
        *t->out_values_read = carquet_column_read_batch(
            t->col_reader, t->dest, t->max_values, NULL, NULL);
    } else {
        *t->out_values_read = 0;
    }
}

/**
 * Fill pipeline slots by reading entire column chunks in parallel.
 * Each task reads ALL values from one column in one row group.
 */
static bulk_read_arg_t pipeline_task_args[512]; /* Max tasks in flight */

static void pipeline_fill(carquet_batch_reader_t* br) {
    if (!br->pipeline_active || !br->pool) return;

    while (br->pipeline_count < br->pipeline_depth &&
           br->rg_order_next < br->rg_order_len) {

        int32_t slot_idx = (br->pipeline_head + br->pipeline_count) % br->pipeline_depth;
        rg_slot_t* slot = &br->pipeline[slot_idx];
        int32_t target_rg = br->rg_order[br->rg_order_next];

        /* Get row count for this RG */
        const parquet_row_group_t* rg = &br->reader->metadata.row_groups[target_rg];
        int64_t rg_rows = rg->num_rows;

        /* Ensure column readers exist and are reset for this slot */
        carquet_error_t err = CARQUET_ERROR_INIT;
        for (int32_t i = 0; i < br->num_projected; i++) {
            int32_t file_col_idx = br->projected_columns[i];
            if (slot->col_readers[i]) {
                reset_column_reader_for_row_group(
                    slot->col_readers[i], br->reader,
                    target_rg, file_col_idx);
            } else {
                slot->col_readers[i] = carquet_reader_get_column(
                    br->reader, target_rg, file_col_idx, &err);
                if (!slot->col_readers[i]) {
                    return;
                }
            }

            /* Ensure value buffer is large enough */
            size_t needed = (size_t)rg_rows * br->projected_value_sizes[i];
            if (needed > slot->col_buf_sizes[i]) {
                free(slot->col_values[i]);
                slot->col_values[i] = malloc(needed);
                slot->col_buf_sizes[i] = slot->col_values[i] ? needed : 0;
                if (!slot->col_values[i]) return;
            }
        }

        slot->rg_index = target_rg;
        slot->ready = false;
        slot->total_rows = rg_rows;
        slot->rows_consumed = 0;

        /* Submit one bulk-read task per column.
         * Each task reads ALL values (all pages) from the column chunk. */
        int32_t base = br->pipeline_count * br->num_projected;
        for (int32_t i = 0; i < br->num_projected; i++) {
            int32_t tidx = base + i;
            if (tidx >= 512) break; /* Safety limit */
            pipeline_task_args[tidx].col_reader = slot->col_readers[i];
            pipeline_task_args[tidx].dest = slot->col_values[i];
            pipeline_task_args[tidx].max_values = rg_rows;
            pipeline_task_args[tidx].out_values_read = &slot->col_num_values[i];
            carquet_worker_pool_submit(br->pool, bulk_read_task,
                                       &pipeline_task_args[tidx]);
        }

        br->pipeline_count++;
        br->rg_order_next++;
    }
}

carquet_status_t carquet_batch_reader_next(
    carquet_batch_reader_t* batch_reader,
    carquet_row_batch_t** batch) {

    /* batch_reader and batch are nonnull per API contract */
    carquet_error_t err = CARQUET_ERROR_INIT;

    /* ====================================================================
     * PIPELINE FAST PATH: serve pre-read data directly from ring buffer
     * ====================================================================
     * When pipeline is active, ALL column data has been bulk-read into
     * contiguous buffers by worker pool threads. We just memcpy batches
     * from those buffers. No column readers, no per-page overhead. */
    if (batch_reader->pipeline_active) {
        /* Check if we need to advance to the next pipeline slot */
        rg_slot_t* slot = NULL;
        if (batch_reader->pipeline_count > 0) {
            slot = &batch_reader->pipeline[batch_reader->pipeline_head];
            if (slot->rows_consumed >= slot->total_rows) {
                /* Current slot exhausted — retire it and advance */
                slot->rg_index = -1;
                batch_reader->pipeline_head = (batch_reader->pipeline_head + 1) % batch_reader->pipeline_depth;
                batch_reader->pipeline_count--;
                slot = NULL;
            }
        }

        if (!slot) {
            /* Fill and wait for new pipeline slots */
            pipeline_fill(batch_reader);
            if (batch_reader->pipeline_count == 0) {
                *batch = NULL;
                return CARQUET_ERROR_END_OF_DATA;
            }
            carquet_worker_pool_wait(batch_reader->pool);
            slot = &batch_reader->pipeline[batch_reader->pipeline_head];

            /* Refill freed slots for next round */
            pipeline_fill(batch_reader);
        }

        /* Reuse or allocate batch struct */
        carquet_row_batch_t* new_batch = batch_reader->cached_batch;
        if (!new_batch) {
            new_batch = calloc(1, sizeof(carquet_row_batch_t));
            if (!new_batch) return CARQUET_ERROR_OUT_OF_MEMORY;
            if (carquet_arena_init(&new_batch->arena) != CARQUET_OK) {
                free(new_batch);
                return CARQUET_ERROR_OUT_OF_MEMORY;
            }
            new_batch->columns = carquet_arena_calloc(&new_batch->arena,
                batch_reader->num_projected, sizeof(carquet_column_data_t));
            if (!new_batch->columns) {
                carquet_arena_destroy(&new_batch->arena);
                free(new_batch);
                return CARQUET_ERROR_OUT_OF_MEMORY;
            }
            new_batch->pooled = true;
            batch_reader->cached_batch = new_batch;
        }

        memset(new_batch->columns, 0, sizeof(carquet_column_data_t) * batch_reader->num_projected);
        new_batch->num_columns = batch_reader->num_projected;

        int64_t remaining = slot->total_rows - slot->rows_consumed;
        int64_t rows_to_read = remaining > batch_reader->config.batch_size
                             ? batch_reader->config.batch_size : remaining;

        /* Copy data from pre-read buffers into batch (zero-copy view) */
        for (int32_t i = 0; i < batch_reader->num_projected; i++) {
            carquet_column_data_t* col = &new_batch->columns[i];
            size_t vs = batch_reader->projected_value_sizes[i];
            size_t offset = (size_t)slot->rows_consumed * vs;

            col->data = (uint8_t*)slot->col_values[i] + offset;
            col->num_values = rows_to_read;
            col->type = batch_reader->projected_types[i];
            col->type_length = batch_reader->projected_type_lengths[i];
            col->ownership = CARQUET_DATA_VIEW;
            col->null_bitmap = NULL;  /* REQUIRED columns — no nulls */
        }

        slot->rows_consumed += rows_to_read;
        new_batch->num_rows = rows_to_read;
        batch_reader->total_rows_read += rows_to_read;
        *batch = new_batch;
        return CARQUET_OK;
    }

    /* ====================================================================
     * SEQUENTIAL PATH (non-mmap, uncompressed, or single RG)
     * ==================================================================== */

    /* Check if we need to move to next row group */
    if (batch_reader->current_row_group < 0 ||
        !carquet_column_has_next(batch_reader->col_readers[0])) {

        int32_t num_row_groups = carquet_reader_num_row_groups(batch_reader->reader);
        batch_reader->current_row_group++;
        if (batch_reader->current_row_group >= num_row_groups) {
            *batch = NULL;
            return CARQUET_ERROR_END_OF_DATA;
        }

        /* Apply row group filter */
        while (batch_reader->config.row_group_filter) {
            bool should_read = batch_reader->config.row_group_filter(
                batch_reader->reader, batch_reader->current_row_group,
                batch_reader->config.row_group_filter_ctx);
            if (should_read) break;
            batch_reader->current_row_group++;
            if (batch_reader->current_row_group >= num_row_groups) {
                *batch = NULL;
                return CARQUET_ERROR_END_OF_DATA;
            }
        }

        carquet_status_t status = open_row_group_readers(
            batch_reader, batch_reader->current_row_group, &err);
        if (status != CARQUET_OK) {
            return status;
        }
    }

    /* Reuse or allocate batch struct */
    carquet_row_batch_t* new_batch = batch_reader->cached_batch;
    if (!new_batch) {
        new_batch = calloc(1, sizeof(carquet_row_batch_t));
        if (!new_batch) {
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        if (carquet_arena_init(&new_batch->arena) != CARQUET_OK) {
            free(new_batch);
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        new_batch->columns = carquet_arena_calloc(&new_batch->arena,
            batch_reader->num_projected, sizeof(carquet_column_data_t));
        if (!new_batch->columns) {
            carquet_arena_destroy(&new_batch->arena);
            free(new_batch);
            return CARQUET_ERROR_OUT_OF_MEMORY;
        }
        new_batch->pooled = true;
        batch_reader->cached_batch = new_batch;
    }

    /* Reset column data for this batch */
    memset(new_batch->columns, 0, sizeof(carquet_column_data_t) * batch_reader->num_projected);
    new_batch->num_columns = batch_reader->num_projected;

    int64_t batch_size = batch_reader->config.batch_size;
    int64_t rows_to_read = carquet_column_remaining(batch_reader->col_readers[0]);
    if (rows_to_read > batch_size) {
        rows_to_read = batch_size;
    }

    /* Handle empty row group - return empty batch, not an error */
    if (rows_to_read == 0) {
        new_batch->num_rows = 0;
        *batch = new_batch;
        return CARQUET_OK;
    }

    /* Read each column - potentially in parallel */
    bool read_error = false;

    /* Uncompressed fixed-width mmap columns can often be served entirely as
     * direct page views. Pre-load those pages serially before the parallel
     * decision so we can clamp the batch to the current page window and avoid
     * both the zero-byte peek copy path and the column-parallel barrier cost. */
    {
        bool zero_copy_candidates = true;
        for (int32_t zi = 0; zi < batch_reader->num_projected; zi++) {
            if (!column_is_zero_copy_candidate(
                    batch_reader->col_readers[zi],
                    batch_reader->projected_types[zi],
                    batch_reader->projected_max_defs[zi])) {
                zero_copy_candidates = false;
                break;
            }
        }

        if (zero_copy_candidates) {
            for (int32_t zi = 0; zi < batch_reader->num_projected; zi++) {
                carquet_column_reader_t* col_reader = batch_reader->col_readers[zi];
                if (col_reader && col_reader->values_remaining > 0) {
                    carquet_status_t status = carquet_column_ensure_page_loaded(col_reader, &err);
                    if (status != CARQUET_OK) {
                        return status;
                    }
                }
            }
        }
    }

    /* ========================================================================
     * PARALLEL PAGE PREFETCH PHASE
     * ========================================================================
     * Pre-load pages for ALL columns in parallel BEFORE reading.
     * Uses persistent worker pool (no per-batch fork/join overhead) when
     * available, falls back to OpenMP, then serial.
     *
     * Only parallelize for mmap (fread is not thread-safe) and when there
     * are columns needing decompression (uncompressed pages are trivial). */
    bool is_mmap = (batch_reader->reader->mmap_data != NULL);
    bool needs_decompression = false;
    for (int32_t pi = 0; pi < batch_reader->num_projected; pi++) {
        carquet_column_reader_t* cr = batch_reader->col_readers[pi];
        if (cr && cr->col_meta->codec != CARQUET_COMPRESSION_UNCOMPRESSED) {
            needs_decompression = true;
            break;
        }
    }

#ifdef _OPENMP
    {
        int num_threads = batch_reader->config.num_threads;
        if (num_threads <= 0) num_threads = omp_get_max_threads();
        if (num_threads > batch_reader->num_projected) num_threads = batch_reader->num_projected;
        if (num_threads < 1) num_threads = 1;

        int32_t omp_i;
        #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1) if(is_mmap && needs_decompression && num_threads > 1)
        for (omp_i = 0; omp_i < batch_reader->num_projected; omp_i++) {
            carquet_column_reader_t* col_reader = batch_reader->col_readers[omp_i];
            if (col_reader && !col_reader->page_loaded && col_reader->values_remaining > 0) {
                (void)carquet_column_read_batch(col_reader, NULL, 0, NULL, NULL);
            }
        }
    }
#else
    for (int32_t pi = 0; pi < batch_reader->num_projected; pi++) {
        carquet_column_reader_t* col_reader = batch_reader->col_readers[pi];
        if (col_reader && !col_reader->page_loaded && col_reader->values_remaining > 0) {
            (void)carquet_column_read_batch(col_reader, NULL, 0, NULL, NULL);
        }
    }
#endif

    /* If every projected column is backed by a direct page view, trim the
     * batch to the smallest currently available page slice. This avoids
     * copying across page boundaries and lets the main read phase stay on
     * the zero-copy path even when the requested batch size is larger than
     * an individual page. */
    {
        int64_t zero_copy_rows = clamp_rows_to_zero_copy_window(batch_reader, rows_to_read);
        if (zero_copy_rows > 0 && zero_copy_rows < rows_to_read) {
            rows_to_read = zero_copy_rows;
        }
    }

    /* ========================================================================
     * MAIN COLUMN READING PHASE
     * ========================================================================
     * Read from pre-loaded pages. Since pages are already decompressed,
     * this phase is mostly memory copies / zero-copy pointer setup.
     * Uses pooled buffers to avoid per-batch malloc/free.
     *
     * Worker pool is used for parallel column reading when pool is available
     * and columns need non-trivial work. Otherwise serial (which is often
     * optimal for zero-copy columns where read_projected_column is ~free).
     */
    bool all_zero_copy_ready = true;
    for (int32_t zi = 0; zi < batch_reader->num_projected; zi++) {
        if (!column_can_zero_copy_batch(
                batch_reader->col_readers[zi],
                batch_reader->projected_types[zi],
                batch_reader->projected_max_defs[zi],
                rows_to_read)) {
            all_zero_copy_ready = false;
            break;
        }
    }

    int32_t col_i;
#ifdef _OPENMP
    {
        int num_threads_read = batch_reader->config.num_threads;
        if (num_threads_read <= 0) num_threads_read = omp_get_max_threads();
        if (num_threads_read > batch_reader->num_projected) num_threads_read = batch_reader->num_projected;
        if (num_threads_read < 1) num_threads_read = 1;

        bool can_par = is_mmap && (num_threads_read > 1) &&
                       (batch_reader->num_projected > 1) && !all_zero_copy_ready;
        if (can_par) {
            #pragma omp parallel for num_threads(num_threads_read) schedule(dynamic, 1)
            for (col_i = 0; col_i < batch_reader->num_projected; col_i++) {
                read_projected_column(batch_reader, new_batch, col_i, rows_to_read, &read_error);
            }
        } else {
            for (col_i = 0; col_i < batch_reader->num_projected; col_i++) {
                read_projected_column(batch_reader, new_batch, col_i, rows_to_read, &read_error);
            }
        }
    }
#else
    for (col_i = 0; col_i < batch_reader->num_projected; col_i++) {
        read_projected_column(batch_reader, new_batch, col_i, rows_to_read, &read_error);
    }
#endif

    if (read_error) {
        /* Don't free cached_batch, just return error */
        return CARQUET_ERROR_DECODE;
    }

    new_batch->num_rows = new_batch->columns[0].num_values;
    batch_reader->total_rows_read += new_batch->num_rows;

    *batch = new_batch;
    return CARQUET_OK;
}

void carquet_batch_reader_free(carquet_batch_reader_t* batch_reader) {
    if (!batch_reader) return;

    /* Drain any in-flight pipeline tasks before freeing */
    if (batch_reader->pool) {
        carquet_worker_pool_wait(batch_reader->pool);
    }

    /* Free pipeline slots */
    if (batch_reader->pipeline) {
        for (int32_t s = 0; s < batch_reader->pipeline_depth; s++) {
            rg_slot_t* slot = &batch_reader->pipeline[s];
            if (slot->col_readers) {
                for (int32_t i = 0; i < batch_reader->num_projected; i++) {
                    if (slot->col_readers[i]) {
                        carquet_column_reader_free(slot->col_readers[i]);
                    }
                }
                free(slot->col_readers);
            }
            if (slot->col_values) {
                for (int32_t i = 0; i < batch_reader->num_projected; i++) {
                    free(slot->col_values[i]);
                }
                free(slot->col_values);
            }
            free(slot->col_buf_sizes);
            free(slot->col_num_values);
        }
        free(batch_reader->pipeline);
    }

    /* Destroy worker pool */
    carquet_worker_pool_destroy(batch_reader->pool);

    /* Free column readers */
    if (batch_reader->col_readers) {
        for (int32_t i = 0; i < batch_reader->num_projected; i++) {
            if (batch_reader->col_readers[i]) {
                carquet_column_reader_free(batch_reader->col_readers[i]);
            }
        }
        free(batch_reader->col_readers);
    }

    /* Free buffer pools */
    if (batch_reader->col_pools) {
        for (int32_t i = 0; i < batch_reader->num_projected; i++) {
            free(batch_reader->col_pools[i].data);
            free(batch_reader->col_pools[i].null_bitmap);
            free(batch_reader->col_pools[i].def_levels);
        }
        free(batch_reader->col_pools);
    }

    /* Free cached batch struct (but NOT pool buffers - those are freed above) */
    if (batch_reader->cached_batch) {
        carquet_arena_destroy(&batch_reader->cached_batch->arena);
        free(batch_reader->cached_batch);
    }

    free(batch_reader->rg_order);
    free(batch_reader->projected_value_sizes);
    free(batch_reader->projected_max_defs);
    free(batch_reader->projected_type_lengths);
    free(batch_reader->projected_types);
    free(batch_reader->projected_columns);
    free(batch_reader);
}

/* ============================================================================
 * Row Batch Implementation
 * ============================================================================
 */

int64_t carquet_row_batch_num_rows(const carquet_row_batch_t* batch) {
    /* batch is nonnull per API contract */
    return batch->num_rows;
}

int32_t carquet_row_batch_num_columns(const carquet_row_batch_t* batch) {
    /* batch is nonnull per API contract */
    return batch->num_columns;
}

carquet_status_t carquet_row_batch_column(
    const carquet_row_batch_t* batch,
    int32_t column_index,
    const void** data,
    const uint8_t** null_bitmap,
    int64_t* num_values) {

    /* batch, data, null_bitmap, num_values are nonnull per API contract */
    if (column_index < 0 || column_index >= batch->num_columns) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    const carquet_column_data_t* col = &batch->columns[column_index];

    *data = col->data;
    *null_bitmap = col->null_bitmap;
    *num_values = col->num_values;

    return CARQUET_OK;
}

carquet_status_t carquet_row_batch_column_dictionary(
    const carquet_row_batch_t* batch,
    int32_t column_index,
    const uint32_t** indices,
    const uint8_t** null_bitmap,
    int64_t* num_values,
    const uint8_t** dictionary_data,
    int32_t* dictionary_count,
    const uint32_t** dictionary_offsets) {

    if (column_index < 0 || column_index >= batch->num_columns) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    const carquet_column_data_t* col = &batch->columns[column_index];

    if (!col->is_dictionary) {
        return CARQUET_ERROR_INVALID_ARGUMENT;
    }

    *indices = (const uint32_t*)col->data;
    *null_bitmap = col->null_bitmap;
    *num_values = col->num_values;
    *dictionary_data = col->dictionary_data;
    *dictionary_count = col->dictionary_count;
    if (dictionary_offsets) {
        *dictionary_offsets = col->dictionary_offsets;
    }

    return CARQUET_OK;
}

void carquet_row_batch_free(carquet_row_batch_t* batch) {
    if (!batch) return;

    /* Pooled batches are owned by the batch_reader - don't free data */
    if (batch->pooled) {
        /* Data buffers belong to the batch_reader's pool.
         * The batch struct itself is cached and reused.
         * This is a no-op - the caller should just drop the pointer. */
        return;
    }

    /* Non-pooled batch: free column data (only if owned, not views into mmap) */
    for (int32_t i = 0; i < batch->num_columns; i++) {
        if (batch->columns[i].ownership == CARQUET_DATA_OWNED) {
            free(batch->columns[i].data);
        }
        free(batch->columns[i].null_bitmap);
    }

    carquet_arena_destroy(&batch->arena);
    free(batch);
}
