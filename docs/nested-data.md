# Nested and Nullable Data

## The Mental Model

Carquet gives you direct Parquet semantics:

- schema tree
- leaf columns
- definition levels
- repetition levels

There is no row-object layer in the C API. Even for nested data, you still write and read leaf columns plus level streams.

## Prefer the Schema Helpers

Use the helpers when your data matches standard Parquet layouts:

- `carquet_schema_add_group()`
- `carquet_schema_add_list()`
- `carquet_schema_add_map()`

Example:

```c
carquet_schema_t* schema = carquet_schema_create(NULL);
carquet_schema_add_column(schema, "id",
    CARQUET_PHYSICAL_INT32, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);

int32_t scores_group = carquet_schema_add_list(schema, "scores",
    CARQUET_PHYSICAL_INT32, NULL, CARQUET_REPETITION_OPTIONAL, 0, 0);

int32_t tags_group = carquet_schema_add_map(schema, "tags",
    CARQUET_PHYSICAL_BYTE_ARRAY, NULL, 0,
    CARQUET_PHYSICAL_BYTE_ARRAY, NULL, 0,
    CARQUET_REPETITION_OPTIONAL, 0);
```

These helpers build the standard 3-level Parquet encodings for LIST and MAP. That keeps the schema interoperable with Arrow, DuckDB, Spark, and other readers.

## Definition Levels

Definition levels answer "how much of this path exists?"

Common cases:

- required scalar: no `def_levels`
- optional scalar: level at max definition means present, lower means null
- nested optional structures: higher levels mean more of the path is defined

Useful schema queries:

- `carquet_schema_max_def_level()`
- `carquet_schema_node_max_def_level()`

For flat optional columns, the pattern is simple:

```c
int16_t def_levels[] = {1, 0, 1, 1, 0};
```

For lists and maps, compute the exact max levels from the schema rather than hard-coding them.

## Repetition Levels

Repetition levels answer "did this value continue the current repeated container, or start a new one?"

Useful schema queries:

- `carquet_schema_max_rep_level()`
- `carquet_schema_node_max_rep_level()`

When reading repeated data, ask for `rep_levels` from `carquet_column_read_batch()` and reconstruct list boundaries from them.

## Reconstruct Lists While Reading

Top-level `LIST` columns created with `carquet_schema_add_list()` usually use `list_rep_level = 1`.

```c
int32_t values[256];
int16_t rep_levels[256];
int64_t offsets[257];

int64_t n = carquet_column_read_batch(col, values, 256, NULL, rep_levels);
int64_t num_lists = carquet_list_offsets(rep_levels, n, 1, offsets, 257);

for (int64_t i = 0; i < num_lists; i++) {
    int64_t begin = offsets[i];
    int64_t end = offsets[i + 1];
    /* values[begin..end-1] belongs to list i */
}
```

Helpers:

- `carquet_count_rows()`: count top-level logical rows from repetition levels
- `carquet_list_offsets()`: build Arrow-style offsets for one repeated level

## Writing Repeated Data

There is no high-level "append one list" writer. You write the leaf values plus matching `def_levels` and `rep_levels`.

That means:

1. Build the schema with `add_list()` / `add_map()` or explicit groups.
2. Query max def/rep levels from the schema.
3. Emit one leaf stream per column with Parquet-correct levels.

If your application already has Arrow-style offsets, convert them to repetition levels first, then call `carquet_writer_write_batch()`.

## Schema Introspection

Use schema introspection when you need to validate generated level streams or map projected columns back to paths:

- `carquet_schema_find_column()`
- `carquet_schema_column_name()`
- `carquet_schema_column_path()`
- `carquet_schema_get_element()`
- `carquet_schema_node_name()`
- `carquet_schema_node_is_leaf()`
- `carquet_schema_node_physical_type()`
- `carquet_schema_node_logical_type()`
- `carquet_schema_node_repetition()`
- `carquet_schema_node_type_length()`

This is especially useful for generic readers and schema-driven code generation.
