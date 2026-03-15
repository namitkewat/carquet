# Changelog

## v0.4.0

### New Features

- **CLI tool (`carquet`)**: Ships a built-in command-line tool for inspecting Parquet files and generating reader code. Built by default (`CARQUET_BUILD_CLI=ON`), installed globally with `make install`.
  - `schema` — print file schema
  - `info` — print detailed file metadata
  - `head` / `tail` — print first/last N rows
  - `count` — print total row count
  - `columns` — list column names (one per line)
  - `stat` — print column statistics
  - `validate` — verify file integrity
  - `sample` — print N random rows
  - `codegen` — generate C reader code from a Parquet file's schema
  - All subcommands support `-h` / `--help`.

- **Code generation (`carquet codegen`)**: Reads a real Parquet file's schema and generates a complete, compilable C program tailored to that schema.
  - `-f` / `--file` — input Parquet file to inspect (generates a placeholder path if omitted)
  - `-o` / `--output` — output source file (prints compile command on stderr)
  - `-b` / `--batch-size` — batch size in generated code
  - `-c` / `--columns` — comma-separated column filter
  - `--mmap` — generate memory-mapped I/O reader
  - `--skeleton` — generate empty `process_batch` body for custom logic
  - Auto-detects compiler (respects `$CC`), carquet include/lib paths, and link dependencies
  - Embeds the source Parquet file as default input so the generated binary works without arguments
  - Generated code compiles with zero warnings

- **Versioned manual in `docs/`**: Added focused in-repo documentation for the main workflows and API concepts.
  - `docs/README.md` — manual index and API surface guide
  - `docs/reading.md` — reader setup, batch scans, column reads, filtering, metadata inspection
  - `docs/writing.md` — schema creation, required/nullable writes, row groups, buffer writer
  - `docs/nested-data.md` — groups, lists, maps, definition levels, repetition levels
  - `docs/performance.md` — mmap, zero-copy, dictionary-preserving reads, prebuffering, tuning
  - `docs/error-handling.md` — status codes, rich error context, type mapping, and level/null conventions

- **Row group predicate pushdown in batch reader**: Added `row_group_filter` callback to `carquet_batch_reader_config_t` for zero-I/O elimination of non-matching row groups using column statistics.
- **I/O coalescing**: Added `carquet_reader_prebuffer()` to pre-read multiple column chunks in a single coalesced read.
- **Speculative footer read**: File open reads up to 64KB from the end in a single I/O call, reducing the open path from 3 I/O calls to 2 for most files.
- **Data Page V2 decoding**: Page reader support for Parquet Data Page V2.
- **Write-path profiling target**: Added the `profile_write` binary for dedicated write-path profiling.

### New APIs

- **Bloom filter query**: Read bloom filters from Parquet files and check value membership — enables column-chunk-level predicate pushdown.
  `carquet_reader_get_bloom_filter()`, `carquet_bloom_filter_check_i32/i64/float/double/bytes()`, `carquet_bloom_filter_size()`, `carquet_bloom_filter_destroy()`
- **Page index read**: Access per-page min/max statistics (column index) and page file locations (offset index) — enables page-level predicate pushdown, skipping individual pages within a column chunk.
  `carquet_reader_get_column_index()`, `carquet_column_index_num_pages()`, `carquet_column_index_get_page_stats()`, `carquet_reader_get_offset_index()`, `carquet_offset_index_get_page_location()`
- **Key-value metadata**: Read and write arbitrary string key-value pairs in the Parquet footer (used by Pandas, Arrow, Spark for schema annotations).
  `carquet_reader_num_metadata()`, `carquet_reader_get_metadata()`, `carquet_reader_find_metadata()`, `carquet_writer_add_metadata()`
- **Column chunk metadata**: Inspect per-column-per-row-group details: codec, encoding, sizes, and which optional features (bloom filter, page index) are present.
  `carquet_reader_column_chunk_metadata()`
- **Per-column writer options**: Override global encoding, compression, statistics, and bloom filter settings on a per-column basis.
  `carquet_writer_set_column_encoding()`, `carquet_writer_set_column_compression()`, `carquet_writer_set_column_statistics()`, `carquet_writer_set_column_bloom_filter()`
- **Buffer writer**: Write Parquet data to an in-memory buffer instead of a file — useful for network protocols, embedding, and testing.
  `carquet_writer_create_buffer()`, `carquet_writer_get_buffer()`

### Performance

- **Multi-row-group pipeline decompression**: Persistent worker pool with pipeline ring buffer for parallel bulk-reads. On 10M-row / 10-RG / 3-column benchmark (Apple M3): snappy 16ms (was 40ms), zstd 25ms (was 44ms), lz4 12ms (was 26ms).
- **ZSTD thread safety**: Per-thread `ZSTD_DCtx`/`ZSTD_CCtx` cache via `pthread_key_create` on all POSIX builds.
- **ARM NEON byte-stream-split**: Widened AArch64 double encode/decode hot loop to 4 doubles at a time.
- **Cheaper page-load peeks**: Zero-length batch reads share the page-loader helper.

### Bug Fixes

- **Fixed BYTE_ARRAY nullable column reads**: Writer now encodes all `num_values` entries for BYTE_ARRAY columns (including zero-length entries for nulls) so values stay aligned with definition levels. Reader's PLAIN decoder and dictionary lookup paths updated to match.
- **Fixed dictionary-encoded nullable columns**: Dictionary index decoding and lookup now process `num_values` entries instead of `non_null_count`.

### Internal

- CLI sources in `src/cli/`: `main.c`, `commands.c`, `codegen.c`, `codegen_read.c`, `codegen_write.c` (stub).
- Build option `CARQUET_BUILD_CLI` (default ON), installs `carquet` binary alongside the library.
- Batch reader pipeline serves pre-read data via zero-copy views from ring buffer slots.
- Worker pool queue capacity increased to 512 for cross-RG bulk-read task submission.
- Page reader fread path uses `prebuf_read_at()` helper for prebuffer cache.
- Windows compatibility: `_getcwd`, `_access`, `_fullpath`, `gmtime_s` behind `#ifdef _WIN32`.

## v0.3.1

- Snappy compression updates and fuzzing improvements.

## v0.3.1_2

- Minor build fix.

## v0.3.1_1

- Minor build fix.

## v0.3.0_6

- Windows build fixes.
