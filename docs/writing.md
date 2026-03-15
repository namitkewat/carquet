# Writing Files

## Build the Schema First

The writer is column-oriented. Define the schema, then write one leaf column at a time.

```c
carquet_error_t err = CARQUET_ERROR_INIT;
carquet_schema_t* schema = carquet_schema_create(&err);
if (!schema) {
    fprintf(stderr, "%s\n", err.message);
    return 1;
}

carquet_logical_type_t string_type = { .id = CARQUET_LOGICAL_STRING };

carquet_schema_add_column(schema, "id",
    CARQUET_PHYSICAL_INT64, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
carquet_schema_add_column(schema, "name",
    CARQUET_PHYSICAL_BYTE_ARRAY, &string_type, CARQUET_REPETITION_REQUIRED, 0, 0);
carquet_schema_add_column(schema, "score",
    CARQUET_PHYSICAL_DOUBLE, NULL, CARQUET_REPETITION_OPTIONAL, 0, 0);
```

For nested schemas, see [`nested-data.md`](./nested-data.md).

## Create the Writer

```c
carquet_writer_options_t opts;
carquet_writer_options_init(&opts);
opts.compression = CARQUET_COMPRESSION_ZSTD;
opts.compression_level = 1;
opts.row_group_size = 128 * 1024 * 1024;
opts.page_size = 1 * 1024 * 1024;
opts.write_statistics = true;
opts.write_crc = true;
opts.write_page_index = true;
opts.write_bloom_filters = true;

carquet_writer_t* writer = carquet_writer_create("out.parquet", schema, &opts, &err);
if (!writer) {
    fprintf(stderr, "%s\n", err.message);
    carquet_schema_free(schema);
    return 1;
}

carquet_schema_free(schema);  /* Safe after writer creation */
```

Other writer entry points:

- `carquet_writer_create_file(FILE*, ...)`
- `carquet_writer_create_buffer(...)`

## Write Required Columns

Required columns are straightforward: one value per logical row, no level arrays.

```c
int64_t ids[] = {1, 2, 3};
carquet_byte_array_t names[] = {
    {(uint8_t*)"alice", 5},
    {(uint8_t*)"bob", 3},
    {(uint8_t*)"carol", 5},
};

carquet_writer_write_batch(writer, 0, ids, 3, NULL, NULL);
carquet_writer_write_batch(writer, 1, names, 3, NULL, NULL);
```

## Write Nullable Columns Correctly

This is the rule most users get wrong:

- `num_values` is the logical row count.
- `def_levels` has one entry per logical row.
- `values` contains only the present values, packed contiguously.

Example for logical rows `[1.5, NULL, 3.5, NULL, 5.5]`:

```c
double values[] = {1.5, 3.5, 5.5};
int16_t def_levels[] = {1, 0, 1, 0, 1};

carquet_writer_write_batch(writer, 2, values, 5, def_levels, NULL);
```

You can query the required definition level from the schema with `carquet_schema_max_def_level()` when you are generating levels programmatically.

## Row Groups, Metadata, and Per-Column Overrides

Important writer invariants:

- Every column must advance by the same logical row count.
- Call `carquet_writer_new_row_group()` only when all columns are aligned.
- `carquet_writer_close()` has the same requirement.

Useful APIs before the first write:

- `carquet_writer_add_metadata()`: add footer key-value pairs
- `carquet_writer_set_column_encoding()`
- `carquet_writer_set_column_compression()`
- `carquet_writer_set_column_statistics()`
- `carquet_writer_set_column_bloom_filter()`

Per-column overrides must be set after writer creation and before writing data.

## Write to Memory Instead of a File

Using an existing schema, you can write to memory and retrieve the final Parquet bytes after close:

```c
carquet_writer_t* writer = carquet_writer_create_buffer(schema, NULL, &err);
if (!writer) {
    fprintf(stderr, "%s\n", err.message);
    return 1;
}

int32_t xs[] = {10, 20, 30};
carquet_writer_write_batch(writer, 0, xs, 3, NULL, NULL);
carquet_writer_close(writer);

void* buffer = NULL;
size_t size = 0;
if (carquet_writer_get_buffer(writer, &buffer, &size) == CARQUET_OK) {
    /* buffer now belongs to the caller */
    free(buffer);
}
```

This is useful for RPC payloads, tests, and embedding Parquet output inside a larger container format.

## Failure Path

If any write step fails, call `carquet_writer_abort()` unless `carquet_writer_close()` has already succeeded. `abort()` frees writer resources but does not produce a valid Parquet file.
