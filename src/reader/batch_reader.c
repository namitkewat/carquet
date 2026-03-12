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
    void* data;                 /* Column values */
    uint8_t* null_bitmap;       /* Null bitmap (1 bit per value), NULL for REQUIRED */
    int64_t num_values;         /* Number of values */
    size_t data_capacity;       /* Allocated capacity for data */
    carquet_physical_type_t type;
    int32_t type_length;        /* For fixed-length types */
    carquet_data_ownership_t ownership;  /* OWNED or VIEW (for future zero-copy) */
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
        max_def != 0 || type == CARQUET_PHYSICAL_BYTE_ARRAY) {
        return false;
    }

    int32_t page_available = col_reader->page_num_values - col_reader->page_values_read;
    return page_available > 0 && page_available >= (int32_t)rows_to_read;
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

    /* Check if direct page handoff is possible:
     * - Column is REQUIRED (no nulls, no definition levels)
     * - Page loader can expose a stable view (mmap or reusable page buffer)
     * - Entire page slice fits in this batch
     */
    bool try_zero_copy = (max_def == 0) &&
                         (!col_reader->page_loaded);

    if (try_zero_copy) {
        /* Trigger page load to check if it's a zero-copy page */
        int64_t dummy_read = carquet_column_read_batch(
            col_reader, NULL, 0, NULL, NULL);
        (void)dummy_read;
    }

    bool use_zero_copy = column_can_zero_copy_batch(
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
    if (value_size == 0 || rows_to_read <= 0) {
        *read_error = true;
        return;
    }

    /* Check for multiplication overflow (max 1GB allocation) */
    #define CARQUET_MAX_BATCH_ALLOC (1024ULL * 1024 * 1024)
    if (value_size > CARQUET_MAX_BATCH_ALLOC / (size_t)rows_to_read) {
        *read_error = true;
        return;
    }

    size_t data_size = value_size * (size_t)rows_to_read;

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
    col_reader->chunk = &rg->columns[column_index];
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

carquet_status_t carquet_batch_reader_next(
    carquet_batch_reader_t* batch_reader,
    carquet_row_batch_t** batch) {

    /* batch_reader and batch are nonnull per API contract */
    carquet_error_t err = CARQUET_ERROR_INIT;
    int32_t num_row_groups = carquet_reader_num_row_groups(batch_reader->reader);

    /* Check if we need to move to next row group */
    if (batch_reader->current_row_group < 0 ||
        !carquet_column_has_next(batch_reader->col_readers[0])) {

        batch_reader->current_row_group++;
        if (batch_reader->current_row_group >= num_row_groups) {
            *batch = NULL;
            return CARQUET_ERROR_END_OF_DATA;
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

#ifdef _OPENMP
    int num_threads = batch_reader->config.num_threads;
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    if (num_threads > batch_reader->num_projected) {
        num_threads = batch_reader->num_projected;
    }
    if (num_threads < 1) {
        num_threads = 1;
    }

    /* Determine if parallel prefetch is worthwhile.
     * The prefetch phase triggers page loading (including decompression).
     * For uncompressed mmap data, page loading is trivial (just pointer
     * setup), so the OpenMP barrier cost (~10-50us) exceeds the work.
     * For compressed data, parallel decompression is critical for throughput. */
    bool needs_decompression = false;
    for (int32_t pi = 0; pi < batch_reader->num_projected; pi++) {
        carquet_column_reader_t* cr = batch_reader->col_readers[pi];
        if (cr && cr->col_meta->codec != CARQUET_COMPRESSION_UNCOMPRESSED) {
            needs_decompression = true;
            break;
        }
    }

    /* ========================================================================
     * PARALLEL PAGE PREFETCH PHASE
     * ========================================================================
     * Pre-load pages for ALL columns in parallel BEFORE reading.
     * This is critical for ZSTD performance because it exposes decompression
     * work across all projected columns before the copy/zero-copy phase runs.
     * The parallelism here is bounded by the number of projected columns.
     *
     * The if(needs_decompression) clause runs this serially when columns
     * are uncompressed, avoiding ~10-50us of barrier overhead per batch.
     */
    int32_t omp_i;  /* Declared outside for MSVC OpenMP compatibility */
    #pragma omp parallel for num_threads(num_threads) schedule(static) if(needs_decompression && num_threads > 1)
    for (omp_i = 0; omp_i < batch_reader->num_projected; omp_i++) {
        carquet_column_reader_t* col_reader = batch_reader->col_readers[omp_i];
        if (col_reader && !col_reader->page_loaded && col_reader->values_remaining > 0) {
            /* Trigger page load (including decompression) without consuming values.
             * The page will be decompressed into col_reader->decoded_values. */
            (void)carquet_column_read_batch(col_reader, NULL, 0, NULL, NULL);
        }
    }
#endif

    /* ========================================================================
     * MAIN COLUMN READING PHASE
     * ========================================================================
     * Now read from pre-loaded pages. Since pages are already decompressed,
     * this phase is mostly memory copies which are fast.
     * Uses pooled buffers to avoid per-batch malloc/free.
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

    int32_t col_i;  /* Declared outside for MSVC OpenMP compatibility */
#ifdef _OPENMP
    bool can_parallelize_columns = (num_threads > 1) && (batch_reader->num_projected > 1);
    if (can_parallelize_columns && !all_zero_copy_ready) {
        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (col_i = 0; col_i < batch_reader->num_projected; col_i++) {
            read_projected_column(batch_reader, new_batch, col_i, rows_to_read, &read_error);
        }
    } else
#endif
    for (col_i = 0; col_i < batch_reader->num_projected; col_i++) {
        read_projected_column(batch_reader, new_batch, col_i, rows_to_read, &read_error);
    }

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
