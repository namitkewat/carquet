# Reading Files

## Choose the Reader API

Use the batch reader unless you specifically need raw Parquet level streams.

| Need | API |
| --- | --- |
| Project a subset of columns, iterate in row batches, parallelize work | `carquet_batch_reader_t` |
| Read one column from one row group, inspect `def_levels` / `rep_levels`, or skip values manually | `carquet_column_reader_t` |
| Only inspect footer metadata | `carquet_get_file_info()`, `carquet_validate_file()` |

## Open a Reader

```c
#include <carquet/carquet.h>

carquet_error_t err = CARQUET_ERROR_INIT;

carquet_reader_options_t opts;
carquet_reader_options_init(&opts);
opts.use_mmap = true;          /* Good default for local files */
opts.verify_checksums = true;  /* Keep enabled unless you trust the source */
opts.num_threads = 0;          /* Auto */

carquet_reader_t* reader = carquet_reader_open("data.parquet", &opts, &err);
if (!reader) {
    char buf[512];
    carquet_error_format(&err, buf, sizeof(buf));
    fprintf(stderr, "%s\n", buf);
    return 1;
}
```

Other entry points:

- `carquet_reader_open_file(FILE*, ...)`: caller keeps ownership of the `FILE*`
- `carquet_reader_open_buffer(const void* buffer, size_t size, ...)`: buffer must outlive the reader

Useful metadata calls right after open:

- `carquet_reader_schema()`
- `carquet_reader_num_rows()`
- `carquet_reader_num_row_groups()`
- `carquet_reader_num_columns()`
- `carquet_reader_is_mmap()`

## Batch Reader Workflow

This is the default path for scans and analytics.

```c
carquet_batch_reader_config_t cfg;
carquet_batch_reader_config_init(&cfg);

const char* cols[] = {"id", "price"};
cfg.column_names = cols;
cfg.num_column_names = 2;
cfg.batch_size = 65536;
cfg.num_threads = 0;

carquet_batch_reader_t* br = carquet_batch_reader_create(reader, &cfg, &err);
if (!br) {
    fprintf(stderr, "%s\n", err.message);
    carquet_reader_close(reader);
    return 1;
}

carquet_row_batch_t* batch = NULL;
while (carquet_batch_reader_next(br, &batch) == CARQUET_OK && batch) {
    const void* id_data;
    const uint8_t* id_nulls;
    int64_t id_count;

    carquet_row_batch_column(batch, 0, &id_data, &id_nulls, &id_count);
    const int64_t* ids = id_data;

    carquet_row_batch_free(batch);
    batch = NULL;
}

carquet_batch_reader_free(br);
carquet_reader_close(reader);
```

Notes:

- `column_indices` takes precedence over `column_names`.
- `row_group_filter` lets you skip entire row groups before any data pages are read. Use it with `carquet_reader_row_group_matches()` or `carquet_reader_column_statistics()`.
- `carquet_row_batch_num_columns()` is the number of projected columns, not the file-wide total.
- For nullable columns, the null bitmap uses `1` for present and `0` for null:

```c
bool is_null = null_bitmap && !(null_bitmap[i / 8] & (1u << (i % 8)));
```

- `BYTE_ARRAY` data comes back as `const carquet_byte_array_t*`.
- `FIXED_LEN_BYTE_ARRAY` data comes back as tightly packed bytes. Use the schema's type length to stride the buffer correctly.

## Column Reader Workflow

Use a column reader when you need explicit control over row groups or Parquet levels.

```c
carquet_column_reader_t* col = carquet_reader_get_column(reader, 0, 2, &err);
if (!col) {
    fprintf(stderr, "%s\n", err.message);
    carquet_reader_close(reader);
    return 1;
}

int32_t values[1024];
int16_t def_levels[1024];
int64_t n;

while ((n = carquet_column_read_batch(col, values, 1024, def_levels, NULL)) > 0) {
    /* values contains only materialized values for this physical type */
}

carquet_column_reader_free(col);
```

Also available:

- `carquet_column_skip()`
- `carquet_column_has_next()`
- `carquet_column_remaining()`

## Predicate Pushdown and Cheap Inspection

Use row-group statistics before you start reading payload pages:

- `carquet_reader_column_statistics()`: get min/max/null counts for one row group + column
- `carquet_reader_row_group_matches()`: ask whether one row group might satisfy a predicate
- `carquet_reader_filter_row_groups()`: collect all candidate row groups at once

Use footer-only helpers when you do not need to build a full reader:

- `carquet_get_file_info()`
- `carquet_validate_file()`

## Metadata, Bloom Filters, and Page Indexes

Carquet exposes the optional metadata structures that many readers hide:

- Key-value footer metadata: `carquet_reader_num_metadata()`, `carquet_reader_get_metadata()`, `carquet_reader_find_metadata()`
- Bloom filters: `carquet_reader_get_bloom_filter()` plus `carquet_bloom_filter_check_*()`
- Page indexes: `carquet_reader_get_column_index()`, `carquet_reader_get_offset_index()`
- Column chunk metadata: `carquet_reader_column_chunk_metadata()`

Use these when you need diagnostics, custom pruning, or interoperability checks. For a compact end-to-end example, see [`examples/advanced_features.c`](../examples/advanced_features.c).

## Lifetime and Ownership

- Close column readers with `carquet_column_reader_free()`.
- Free each row batch with `carquet_row_batch_free()` before asking for the next one.
- Pointers returned from batch APIs belong to the batch. Do not keep them after freeing the batch.
- Pointers returned from reader metadata APIs belong to the reader. Do not keep them after closing the reader.
