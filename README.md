<p align="center">
  <img src="res/img/carquet_logo.png" alt="Carquet" width="280" />
</p>

<h1 align="center">Carquet</h1>

<p align="center">
  A fast, pure C library for reading and writing Apache Parquet files.
</p>

<p align="center">
  <a href="https://github.com/Vitruves/carquet/actions/workflows/cpp.yml"><img src="https://github.com/Vitruves/carquet/actions/workflows/cpp.yml/badge.svg" alt="Build" /></a>
  <img src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-blue" alt="Platform" />
  <img src="https://img.shields.io/badge/C-C11-blue" alt="C Standard" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License" /></a>
  <br/>
  <img src="https://img.shields.io/badge/SIMD-SSE4.2%20%7C%20AVX%20%7C%20AVX2%20%7C%20AVX--512-red" alt="x86 SIMD" />
  <img src="https://img.shields.io/badge/SIMD-NEON%20%7C%20SVE-orange" alt="ARM SIMD" />
</p>

## Highlights

- **Pure C11** with three external dependencies (zstd, zlib, lz4) -- all auto-fetched by CMake
- **~200KB binary** vs ~50MB+ for Arrow
- **Compressed reads 1.5-2.1x faster** than Arrow C++ on x86 (snappy, zstd, lz4)
- **Writes 1.3-2.5x faster** than Arrow C++ and PyArrow on ARM
- Full Parquet spec: all types, encodings, compression codecs, nested schemas, bloom filters, page indexes
- SIMD-optimized (SSE4.2, AVX2, AVX-512, NEON, SVE) with runtime detection and scalar fallbacks
- PyArrow, DuckDB, Spark compatible out of the box

## Performance

All benchmarks use identical data (deterministic LCG PRNG), identical Parquet settings (no dictionary, BYTE_STREAM_SPLIT for floats, page checksums, mmap reads), trimmed median of 11-51 iterations, with OS page cache purged between write and read phases and cooldown between configurations. Schema: 3 columns (INT64, DOUBLE, INT32). Compared against Arrow C++ 23.0.1 (native C++) and PyArrow 23.0.0/23.0.1 (Python bindings to the same C++ library).

### x86: Intel Xeon D-1531 (Linux)

*12 threads @ 2.7GHz, 32GB RAM, Ubuntu 24.04 -- ZSTD level 1*

#### Compressed reads: the headline result

These numbers reflect full decompression + decoding of every value -- no shortcuts.

| | 100M rows | | 10M rows | | 1M rows | |
|--------|-----------|-------|----------|-------|---------|-------|
| **Codec** | **Carquet** | **vs Arrow C++** | **Carquet** | **vs Arrow C++** | **Carquet** | **vs Arrow C++** |
| snappy | 1939ms | **1.4x** | 147ms | **2.1x** | 17ms | **2.1x** |
| zstd | 1653ms | **1.4x** | 168ms | **1.5x** | 17ms | **1.5x** |
| lz4 | — | — | 68ms | **2.0x** | 6.4ms | **1.7x** |

Carquet reads compressed Parquet **1.5-2.1x faster than Arrow C++** across every codec and dataset size tested. Snappy and LZ4 show the largest advantage (~2x at 10M rows) while ZSTD is consistently ~1.5x faster.

#### Full results vs Arrow C++ (10M rows)

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio | Size |
|-------|--------------|-----------------|---------|-------------|----------------|---------|------|
| none | 855ms | 833ms | 0.97x | **30ms** | 105ms | **3.5x**\* | 190.7MB |
| snappy | 1093ms | 980ms | 0.90x | **147ms** | 301ms | **2.1x** | 125.2MB |
| zstd | **1483ms** | 1569ms | **1.06x** | **168ms** | 254ms | **1.5x** | 95.3MB |
| lz4 | 1526ms | 1527ms | 1.00x | **68ms** | 138ms | **2.0x** | 122.9MB |

\* Uncompressed reads use mmap zero-copy (see note below). Compressed reads involve full decompression and decoding.

#### 100M rows (partial -- PyArrow timed out on zstd/lz4)

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **10.2s** | 16.8s | **1.65x** | **216ms** | 757ms | **3.5x**\* |
| snappy | **13.6s** | 14.9s | **1.09x** | **1939ms** | 2724ms | **1.4x** |
| zstd | **14.7s** | 16.0s | **1.09x** | **1653ms** | 2262ms | **1.4x** |

At 100M rows, Carquet writes faster than Arrow C++ across the board (1.09-1.65x).

<details>
<summary>Full results: 100K and 1M rows</summary>

#### 1M rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **179ms** | 189ms | **1.06x** | **0.20ms** | 6.4ms | 32x\* |
| snappy | 205ms | **85ms** | 0.41x | **17ms** | 35ms | **2.1x** |
| zstd | 178ms | **160ms** | 0.90x | **17ms** | 26ms | **1.5x** |
| lz4 | **138ms** | 152ms | **1.10x** | **6.4ms** | 11ms | **1.7x** |

#### 100K rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | 17.4ms | **16.6ms** | 0.95x | **0.07ms** | 0.91ms | 13x\* |
| snappy | **9.7ms** | 10.7ms | **1.10x** | **1.6ms** | 5.2ms | **3.2x** |
| zstd | **10.3ms** | 14.2ms | **1.38x** | **1.6ms** | 3.1ms | **2.0x** |
| lz4 | **10.0ms** | 10.5ms | **1.05x** | **0.7ms** | 1.2ms | **1.7x** |

\* Zero-copy mmap path

</details>

<details>
<summary>Full results: vs PyArrow</summary>

#### 10M rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **855ms** | 1224ms | **1.43x** | **30ms** | 210ms | **6.9x**\* |
| snappy | 1093ms | **1078ms** | 0.99x | **147ms** | 403ms | **2.8x** |
| zstd | **1483ms** | 1767ms | **1.19x** | **168ms** | 353ms | **2.1x** |
| lz4 | 1526ms | **1051ms** | 0.69x | **68ms** | 279ms | **4.1x** |

#### 1M rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **179ms** | 194ms | **1.09x** | **0.20ms** | 9.2ms | 46x\* |
| snappy | 205ms | **152ms** | 0.74x | **17ms** | 19ms | **1.2x** |
| zstd | 178ms | **161ms** | 0.90x | 17ms | **16ms** | 0.97x |
| lz4 | **138ms** | 145ms | **1.05x** | **6.4ms** | 9.9ms | **1.5x** |

#### 100K rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | 17.4ms | **16.4ms** | 0.94x | **0.07ms** | 2.5ms | 36x\* |
| snappy | **9.7ms** | 9.9ms | **1.02x** | **1.6ms** | 6.9ms | **4.2x** |
| zstd | **10.3ms** | 15.1ms | **1.46x** | **1.6ms** | 7.7ms | **5.0x** |
| lz4 | **10.0ms** | 10.5ms | **1.05x** | **0.7ms** | 1.9ms | **2.7x** |

Carquet reads faster than PyArrow across nearly all codecs and sizes. The sole exception is 1M zstd reads where PyArrow is marginally faster (0.97x).

\* Zero-copy mmap path

</details>

### ARM: Apple M3 (macOS)

*MacBook Air M3, 16GB RAM, macOS 26.2, Arrow C++ 23.0.1, PyArrow 23.0.1 -- `-DCARQUET_NATIVE_ARCH=ON`, ZSTD level 1*

#### 10M rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | 136ms | 132ms | 0.97x | **8.7ms** | 14.2ms | **1.6x**\* |
| snappy | **218ms** | 240ms | **1.10x** | 86ms | **53ms** | 0.62x |
| zstd | **202ms** | 338ms | **1.68x** | **66ms** | 70ms | 1.07x |
| lz4 | **206ms** | 254ms | **1.23x** | **25ms** | 31ms | **1.25x** |

#### 1M rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **9.0ms** | 12.8ms | **1.42x** | **0.04ms** | 0.92ms | 23x\* |
| snappy | **13.6ms** | 24.2ms | **1.78x** | 8.6ms | **4.8ms** | 0.56x |
| zstd | **16.0ms** | 34.6ms | **2.16x** | **6.1ms** | 6.4ms | 1.05x |
| lz4 | **13.5ms** | 25.0ms | **1.86x** | **2.4ms** | 2.5ms | 1.05x |

On ARM, the standout is **write performance**: Carquet is **1.1-2.2x faster than Arrow C++** across all codecs and sizes, with ZSTD writes reaching 2.16x. Reads are competitive -- faster on uncompressed and LZ4, roughly even on ZSTD, slower on snappy (Arrow C++ has stronger snappy decompression on this platform).

<details>
<summary>Full results: vs PyArrow on ARM</summary>

#### 10M rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **136ms** | 187ms | **1.38x** | **8.7ms** | 35ms | **4.0x**\* |
| snappy | **218ms** | 293ms | **1.34x** | 86ms | **46ms** | 0.53x |
| zstd | **202ms** | 398ms | **1.97x** | 66ms | **57ms** | 0.86x |
| lz4 | **206ms** | 307ms | **1.49x** | **25ms** | 39ms | **1.56x** |

#### 1M rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **9.0ms** | 17.5ms | **1.95x** | **0.04ms** | 2.6ms | 65x\* |
| snappy | **13.6ms** | 29.8ms | **2.18x** | 8.6ms | **3.6ms** | 0.42x |
| zstd | **16.0ms** | 40.0ms | **2.50x** | 6.1ms | **4.4ms** | 0.72x |
| lz4 | **13.5ms** | 30.6ms | **2.28x** | **2.4ms** | 3.1ms | **1.30x** |

Carquet writes **1.3-2.5x faster than PyArrow** across the board on ARM. Reads are faster on uncompressed and LZ4; PyArrow is faster on snappy and zstd reads (dictionary encoding advantage on this data shape).

\* Zero-copy mmap path

</details>

### A note on uncompressed read numbers

Uncompressed reads marked with \* use Carquet's **mmap zero-copy path**: for PLAIN-encoded, uncompressed, fixed-size, required columns, the batch reader returns pointers directly into the memory-mapped file with no memcpy. The OS only pages in data the application actually touches. This explains the extreme ratios (10-65x) for uncompressed reads -- Arrow always materializes into its own columnar format regardless.

This is a real API-level advantage for workloads like filtering, sampling, or partial scans. For full sequential scans that touch every value, the effective speedup is lower. **The compressed read numbers (snappy, zstd, lz4) involve full decompression and decoding of every value** and are the most representative measure of end-to-end read throughput.

### Summary

| | x86 (Xeon D-1531) | ARM (Apple M3) |
|---|---|---|
| **Compressed reads** | **1.5-2.1x faster** than Arrow C++ | Even to 1.25x faster (LZ4); 0.6x slower (snappy) |
| **Uncompressed reads** | **3.5x faster**\* (mmap zero-copy) | **1.6x faster**\* |
| **Writes** | Competitive (0.90-1.65x, best at scale) | **1.1-2.2x faster** across the board |
| **File sizes** | Equal or slightly smaller | Equal or slightly smaller |

\* Zero-copy path; see note above.

## Building

### Requirements

- C11 compiler (GCC 4.9+, Clang 3.4+, MSVC 2015+)
- CMake 3.16+
- zstd, zlib, lz4 (auto-fetched if missing)
- OpenMP (optional, for parallel column reading)

### Quick Start

```bash
git clone https://github.com/Vitruves/carquet.git
cd carquet
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CARQUET_BUILD_DEV` | OFF | Build everything (tests, examples, benchmarks) |
| `CARQUET_BUILD_TESTS` | OFF | Build test suite only |
| `CARQUET_BUILD_SHARED` | OFF | Build shared library instead of static |
| `CARQUET_NATIVE_ARCH` | OFF | `-march=native` for max performance |
| `CARQUET_ENABLE_SVE` | OFF | ARM SVE (experimental) |

All x86 SIMD (SSE, AVX, AVX2, AVX-512) and ARM NEON are auto-detected and enabled by default.

<details>
<summary>All build options</summary>

| Option | Default | Description |
|--------|---------|-------------|
| `CARQUET_BUILD_EXAMPLES` | OFF | Build example programs |
| `CARQUET_BUILD_BENCHMARKS` | OFF | Build benchmark and profiling programs |
| `CARQUET_BUILD_ARROW_CPP_BENCHMARK` | OFF | Optional Arrow C++ comparison benchmark |
| `CARQUET_BUILD_INTEROP` | OFF | Build interoperability tests |
| `CARQUET_BUILD_FUZZ` | OFF | Build fuzz targets |
| `CARQUET_ENABLE_SSE` | ON | SSE optimizations (x86, auto-detected) |
| `CARQUET_ENABLE_AVX` | ON | AVX optimizations (x86, auto-detected) |
| `CARQUET_ENABLE_AVX2` | ON | AVX2 optimizations (x86, auto-detected) |
| `CARQUET_ENABLE_AVX512` | ON | AVX-512 optimizations (x86, auto-detected) |
| `CARQUET_ENABLE_NEON` | ON | NEON optimizations (ARM, auto-detected) |

</details>

### Development Build

```bash
cmake -B build -DCARQUET_BUILD_DEV=ON
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

## Usage

### Write a Parquet File

```c
#include <carquet/carquet.h>

int main(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    // Define schema
    carquet_schema_t* schema = carquet_schema_create(&err);
    carquet_schema_add_column(schema, "id",    CARQUET_PHYSICAL_INT64,  NULL, CARQUET_REPETITION_REQUIRED, 0, 0);
    carquet_schema_add_column(schema, "value", CARQUET_PHYSICAL_DOUBLE, NULL, CARQUET_REPETITION_REQUIRED, 0, 0);

    // Configure writer
    carquet_writer_options_t opts;
    carquet_writer_options_init(&opts);
    opts.compression = CARQUET_COMPRESSION_ZSTD;

    // Write
    carquet_writer_t* w = carquet_writer_create("output.parquet", schema, &opts, &err);

    int64_t ids[]    = {1, 2, 3, 4, 5};
    double values[]  = {1.1, 2.2, 3.3, 4.4, 5.5};
    carquet_writer_write_batch(w, 0, ids, 5, NULL, NULL);
    carquet_writer_write_batch(w, 1, values, 5, NULL, NULL);
    carquet_writer_close(w);

    carquet_schema_free(schema);
    return 0;
}
```

### Read a Parquet File

```c
#include <carquet/carquet.h>
#include <stdio.h>

int main(void) {
    carquet_error_t err = CARQUET_ERROR_INIT;

    // Open with mmap for best read performance
    carquet_reader_options_t opts;
    carquet_reader_options_init(&opts);
    opts.use_mmap = true;

    carquet_reader_t* r = carquet_reader_open("output.parquet", &opts, &err);
    if (!r) { printf("Error: %s\n", err.message); return 1; }

    printf("Rows: %lld, Columns: %d\n",
           (long long)carquet_reader_num_rows(r),
           carquet_reader_num_columns(r));

    // Batch reader for efficient iteration
    carquet_batch_reader_config_t cfg;
    carquet_batch_reader_config_init(&cfg);
    cfg.batch_size = 65536;

    carquet_batch_reader_t* br = carquet_batch_reader_create(r, &cfg, &err);
    carquet_row_batch_t* batch = NULL;

    while (carquet_batch_reader_next(br, &batch) == CARQUET_OK && batch) {
        const void* data;
        const uint8_t* nulls;
        int64_t n;
        carquet_row_batch_column(batch, 0, &data, &nulls, &n);
        const int64_t* ids = (const int64_t*)data;
        // process ids[0..n-1] ...
        carquet_row_batch_free(batch);
        batch = NULL;
    }

    carquet_batch_reader_free(br);
    carquet_reader_close(r);
    return 0;
}
```

### Nullable Columns

```c
// Schema with nullable column
carquet_schema_add_column(schema, "name", CARQUET_PHYSICAL_BYTE_ARRAY,
                          NULL, CARQUET_REPETITION_OPTIONAL, 0, 0);

// Write with definition levels (1 = present, 0 = null)
carquet_byte_array_t names[] = {{(uint8_t*)"Alice", 5}, {(uint8_t*)"Bob", 3}};
int16_t def_levels[] = {1, 0, 1};  // Alice, NULL, Bob (3 rows, 2 values)
carquet_writer_write_batch(writer, col, names, 3, def_levels, NULL);
```

### Nested Types (Lists, Maps)

```c
// list<int32>
int32_t list_leaf = carquet_schema_add_list(
    schema, "tags", CARQUET_PHYSICAL_INT32, NULL,
    CARQUET_REPETITION_OPTIONAL, 0, 0);

// map<string, int32>
int32_t map_val = carquet_schema_add_map(
    schema, "props",
    CARQUET_PHYSICAL_BYTE_ARRAY, NULL, 0,   // key: string
    CARQUET_PHYSICAL_INT32, NULL, 0,         // value: int32
    CARQUET_REPETITION_OPTIONAL, 0);

// Write list data: row0=[100,200], row1=NULL, row2=[300]
int32_t vals[] = {100, 200, 300};
int16_t def[]  = {  3,   3,   0,   3};
int16_t rep[]  = {  0,   1,   0,   0};
carquet_writer_write_batch(writer, col, vals, 4, def, rep);
```

### Column Projection

```c
carquet_batch_reader_config_t cfg;
carquet_batch_reader_config_init(&cfg);

// Read only specific columns
const char* names[] = {"id", "timestamp"};
cfg.column_names = names;
cfg.num_column_names = 2;
```

### Compression

| Codec | Enum | Best For |
|-------|------|----------|
| ZSTD | `CARQUET_COMPRESSION_ZSTD` | Best overall (great ratio + speed) |
| LZ4 | `CARQUET_COMPRESSION_LZ4_RAW` | Read-heavy workloads (fastest decompression) |
| Snappy | `CARQUET_COMPRESSION_SNAPPY` | Wide compatibility |
| GZIP | `CARQUET_COMPRESSION_GZIP` | Maximum compatibility with older tools |

```c
opts.compression = CARQUET_COMPRESSION_ZSTD;
opts.compression_level = 1;  // 0 = codec default; ZSTD: 1-22, GZIP: 1-9
```

### Writer Options

```c
carquet_writer_options_t opts;
carquet_writer_options_init(&opts);
opts.compression        = CARQUET_COMPRESSION_ZSTD;
opts.row_group_size     = 128 * 1024 * 1024;  // 128 MB row groups
opts.write_statistics   = true;                // min/max for predicate pushdown
opts.write_crc          = true;                // CRC32 page verification
opts.write_bloom_filters = true;               // bloom filters per column
opts.write_page_index   = true;                // column/offset page indexes
```

### Error Handling

```c
carquet_error_t err = CARQUET_ERROR_INIT;
carquet_reader_t* r = carquet_reader_open("data.parquet", NULL, &err);
if (!r) {
    printf("[%s] %s\n", carquet_status_name(err.code), err.message);
    printf("Hint: %s\n", carquet_error_recovery_hint(err.code));
    return 1;
}
```

All functions return `carquet_status_t` or use `carquet_error_t*` out-parameters. Programming errors (NULL where a valid pointer is required) trigger assertions; runtime errors (bad files, OOM) return error codes.

## Interoperability

Carquet files are fully compatible with PyArrow, DuckDB, Spark, and any Parquet reader:

```python
import pyarrow.parquet as pq
table = pq.read_table("carquet_output.parquet")  # just works
```

```sql
-- DuckDB
SELECT * FROM read_parquet('carquet_output.parquet');
```

Bidirectional interop testing:

```bash
cmake -B build -DCARQUET_BUILD_INTEROP=ON && cmake --build build
python3 interop/run_interop.py
```

## Parquet Feature Support

| Feature | Status |
|---------|--------|
| Physical types | All 8 (BOOLEAN through FIXED_LEN_BYTE_ARRAY) |
| Logical types | STRING, DATE, TIME, TIMESTAMP, DECIMAL, UUID, JSON |
| Encodings | PLAIN, RLE, DICTIONARY, DELTA_BINARY_PACKED, DELTA_LENGTH_BYTE_ARRAY, DELTA_BYTE_ARRAY, BYTE_STREAM_SPLIT |
| Compression | UNCOMPRESSED, SNAPPY, GZIP, LZ4, ZSTD |
| Nested schemas | Groups, lists, maps with definition/repetition levels |
| Bloom filters | Read and write |
| Page indexes | Column index + offset index |
| Statistics | Min/max/null count per column chunk |
| CRC32 | Page-level verification (HW-accelerated on ARM) |
| Memory-mapped I/O | Zero-copy reads for uncompressed PLAIN data |
| Column projection | Read only selected columns |
| OpenMP parallel reads | When available |
| Encryption | Not supported |

## Running Benchmarks

```bash
# Build with max optimizations
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCARQUET_NATIVE_ARCH=ON -DCARQUET_BUILD_DEV=ON
cmake --build build -j$(nproc)

cd build
./benchmark_carquet                     # Carquet standalone
python3 ../benchmark/run_benchmark.py   # Full comparison (+ PyArrow, + Arrow C++)

# Skip 100M-row (xlarge) configs — they write ~2GB files per codec
# and can take 30+ minutes depending on hardware
python3 ../benchmark/run_benchmark.py --skip-xlarge

# Override ZSTD level (default: 1)
CARQUET_BENCH_ZSTD_LEVEL=3 python3 ../benchmark/run_benchmark.py
```

<details>
<summary>Optional Arrow C++ benchmark</summary>

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCARQUET_NATIVE_ARCH=ON \
  -DCARQUET_BUILD_BENCHMARKS=ON \
  -DCARQUET_BUILD_ARROW_CPP_BENCHMARK=ON
cmake --build build -j$(nproc)

# Or point at a custom Arrow install
cmake -B build ... -DCARQUET_ARROW_CPP_ROOT=/path/to/arrow-prefix
```

The Arrow C++ benchmark mirrors Carquet's methodology: same data, row group sizing, no dictionary, page checksums, mmap reads, BYTE_STREAM_SPLIT for floats.

</details>

## API Reference

Full API is in [`include/carquet/carquet.h`](include/carquet/carquet.h). Key types:

| Type | Purpose |
|------|---------|
| `carquet_reader_t` | File reader (open from path, FILE*, or memory buffer) |
| `carquet_writer_t` | File writer |
| `carquet_batch_reader_t` | High-level batch iteration |
| `carquet_schema_t` | Schema definition and introspection |
| `carquet_error_t` | Rich error info (code, message, source location, recovery hint) |

<details>
<summary>Core API functions</summary>

**Reader**
```c
carquet_reader_t* carquet_reader_open(const char* path, const carquet_reader_options_t* opts, carquet_error_t* err);
carquet_reader_t* carquet_reader_open_buffer(const void* buf, size_t size, const carquet_reader_options_t* opts, carquet_error_t* err);
void              carquet_reader_close(carquet_reader_t* reader);
int64_t           carquet_reader_num_rows(const carquet_reader_t* reader);
int32_t           carquet_reader_num_columns(const carquet_reader_t* reader);
```

**Batch Reader**
```c
carquet_batch_reader_t* carquet_batch_reader_create(carquet_reader_t* reader, const carquet_batch_reader_config_t* cfg, carquet_error_t* err);
carquet_status_t        carquet_batch_reader_next(carquet_batch_reader_t* br, carquet_row_batch_t** batch);
carquet_status_t        carquet_row_batch_column(const carquet_row_batch_t* batch, int32_t col, const void** data, const uint8_t** nulls, int64_t* n);
```

**Writer**
```c
carquet_writer_t*  carquet_writer_create(const char* path, const carquet_schema_t* schema, const carquet_writer_options_t* opts, carquet_error_t* err);
carquet_status_t   carquet_writer_write_batch(carquet_writer_t* w, int32_t col, const void* values, int64_t n, const int16_t* def, const int16_t* rep);
carquet_status_t   carquet_writer_close(carquet_writer_t* w);
```

**Schema**
```c
carquet_schema_t* carquet_schema_create(carquet_error_t* err);
carquet_status_t  carquet_schema_add_column(carquet_schema_t* s, const char* name, carquet_physical_type_t type, const carquet_logical_type_t* logical, carquet_field_repetition_t rep, int32_t type_len, int32_t parent);
int32_t           carquet_schema_add_list(carquet_schema_t* s, const char* name, carquet_physical_type_t elem_type, const carquet_logical_type_t* elem_logical, carquet_field_repetition_t rep, int32_t type_len, int32_t parent);
int32_t           carquet_schema_add_map(carquet_schema_t* s, const char* name, carquet_physical_type_t key_type, const carquet_logical_type_t* key_logical, int32_t key_len, carquet_physical_type_t val_type, const carquet_logical_type_t* val_logical, int32_t val_len, carquet_field_repetition_t rep, int32_t parent);
```

**Filtering**
```c
int32_t carquet_reader_filter_row_groups(const carquet_reader_t* reader, int32_t col, carquet_compare_op_t op, const void* value, int32_t value_size, int32_t* matching, int32_t max);
```

</details>

## Project Structure

```
include/carquet/   Public API (carquet.h, types.h, error.h)
src/
  core/            Arena allocator, buffer, bitpack, endian
  encoding/        PLAIN, RLE, DELTA, DICTIONARY, BYTE_STREAM_SPLIT
  compression/     Snappy (internal), GZIP, ZSTD, LZ4 (wrappers)
  thrift/          Thrift compact protocol for Parquet metadata
  simd/            Runtime dispatch + x86 (SSE/AVX2/AVX-512) + ARM (NEON/SVE)
  reader/          File, row group, column, page, batch readers + mmap
  writer/          File, row group, column, page writers
  metadata/        Schema, statistics, bloom filters, page indexes
  util/            CRC32, xxHash
tests/             18 test files
examples/          basic_write_read, data_types, compression_codecs, nullable_columns
benchmark/         Performance benchmarks and comparison tools
```

## License

MIT
