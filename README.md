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
- **Compressed reads 1.05-2.1x faster** than Arrow C++ across x86 and ARM
- **Writes 1.1-2.4x faster** than Arrow C++ across codecs and platforms
- Full Parquet spec: all types, encodings, compression codecs, nested schemas, bloom filters, page indexes
- SIMD-optimized (SSE4.2, AVX2, AVX-512, NEON, SVE) with runtime detection and scalar fallbacks
- PyArrow, DuckDB, Spark compatible out of the box

## Performance

Carquet vs Arrow C++ 23.0.1 at 10M rows (the most representative size). Higher ratio = Carquet faster.

| | x86 (Xeon D-1531) | | ARM (Apple M3) | |
|---|---|---|---|---|
| **Codec** | **Write** | **Read** | **Write** | **Read** |
| snappy | **1.02x** | **2.6x** | **1.10x** | **1.41x** |
| zstd | **1.30x** | **1.5x** | **1.67x** | **1.07x** |
| lz4 | 0.97x | **2.0x** | **1.19x** | **1.26x** |
| none | 1.00x | **3.3x**\* | 0.93x | **1.53x**\* |

\* Uncompressed reads use mmap zero-copy -- see note below.

Compressed reads involve full decompression and decoding of every value, no shortcuts. Carquet reads compressed Parquet **1.07-2.6x faster than Arrow C++** across every codec tested on both platforms, while writes are **1.02-1.67x faster** across most configurations.

<details>
<summary>Benchmark methodology</summary>

All benchmarks use identical data (deterministic LCG PRNG), identical Parquet settings (no dictionary, BYTE_STREAM_SPLIT for floats, page checksums, mmap reads), trimmed median of 11-51 iterations, with OS page cache purged between write and read phases and cooldown between configurations. Schema: 3 columns (INT64, DOUBLE, INT32). Compared against Arrow C++ 23.0.1 (native C++) and PyArrow 23.0.1 (Python bindings to the same C++ library).

**Uncompressed reads** marked with \* use Carquet's **mmap zero-copy path**: for PLAIN-encoded, uncompressed, fixed-size, required columns, the batch reader returns pointers directly into the memory-mapped file with no memcpy. The OS only pages in data the application actually touches. Arrow always materializes into its own columnar format regardless. This is a real API-level advantage for filtering, sampling, or partial scans. **The compressed read numbers are the most representative measure of end-to-end read throughput.**

</details>

<details>
<summary>Full x86 results (Intel Xeon D-1531, Linux)</summary>

*12 threads @ 2.7GHz, 32GB RAM, Ubuntu 24.04 -- ZSTD level 1*

#### 10M rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio | Size |
|-------|--------------|-----------------|---------|-------------|----------------|---------|------|
| none | 864ms | 862ms | 1.00x | **30ms** | 101ms | **3.3x**\* | 190.7MB |
| snappy | **1540ms** | 1577ms | **1.02x** | **113ms** | 300ms | **2.6x** | 125.1MB |
| zstd | **1352ms** | 1751ms | **1.30x** | **173ms** | 257ms | **1.5x** | 95.3MB |
| lz4 | 1595ms | 1541ms | 0.97x | **69ms** | 139ms | **2.0x** | 122.9MB |

#### 1M rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **178ms** | 194ms | **1.09x** | **0.32ms** | 6.2ms | **19x**\* |
| snappy | 182ms | **153ms** | 0.84x | **12ms** | 29ms | **2.5x** |
| zstd | 187ms | **159ms** | 0.85x | **17ms** | 24ms | **1.4x** |
| lz4 | 182ms | **150ms** | 0.82x | **6.5ms** | 11ms | **1.8x** |

#### 100K rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **16.8ms** | 18.4ms | **1.09x** | **0.11ms** | 0.97ms | **8.8x**\* |
| snappy | **9.6ms** | 11.0ms | **1.15x** | **1.2ms** | 4.0ms | **3.3x** |
| zstd | **10.3ms** | 12.5ms | **1.21x** | **1.6ms** | 2.9ms | **1.9x** |
| lz4 | 10.2ms | **9.8ms** | 0.96x | **0.66ms** | 1.0ms | **1.5x** |

#### 10M rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **864ms** | 1835ms | **2.12x** | **30ms** | 213ms | **7.1x**\* |
| snappy | 1540ms | **1341ms** | 0.87x | **113ms** | 372ms | **3.3x** |
| zstd | **1352ms** | 1658ms | **1.23x** | **173ms** | 372ms | **2.2x** |
| lz4 | **1595ms** | 1617ms | **1.01x** | **69ms** | 257ms | **3.7x** |

\* Zero-copy mmap path

</details>

<details>
<summary>Full ARM results (Apple M3, macOS)</summary>

*MacBook Air M3, 16GB RAM, macOS 26.2, Arrow C++ 23.0.1, PyArrow 23.0.1 -- ZSTD level 1*

#### 10M rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio | Size |
|-------|--------------|-----------------|---------|-------------|----------------|---------|------|
| none | 136.4ms | 126.5ms | 0.93x | **8.74ms** | 13.33ms | **1.53x**\* | 190.7MB |
| snappy | **221.6ms** | 243.4ms | **1.10x** | **39.53ms** | 55.83ms | **1.41x** | 125.1MB |
| zstd | **204.8ms** | 341.8ms | **1.67x** | **65.62ms** | 69.90ms | **1.07x** | 95.3MB |
| lz4 | **208.0ms** | 247.5ms | **1.19x** | **24.85ms** | 31.33ms | **1.26x** | 122.9MB |

#### 1M rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **8.85ms** | 12.12ms | **1.37x** | **0.04ms** | 0.96ms | 24x\* |
| snappy | **13.52ms** | 24.61ms | **1.82x** | **3.85ms** | 4.84ms | **1.26x** |
| zstd | **16.43ms** | 33.83ms | **2.06x** | **6.12ms** | 6.44ms | **1.05x** |
| lz4 | **13.77ms** | 24.68ms | **1.79x** | **2.40ms** | 2.52ms | **1.05x** |

#### 100K rows vs Arrow C++

| Codec | Carquet Write | Arrow C++ Write | W ratio | Carquet Read | Arrow C++ Read | R ratio |
|-------|--------------|-----------------|---------|-------------|----------------|---------|
| none | **1.38ms** | 1.54ms | **1.12x** | **0.02ms** | 0.11ms | 5.50x\* |
| snappy | **1.65ms** | 2.59ms | **1.57x** | **0.33ms** | 0.52ms | **1.58x** |
| zstd | **1.79ms** | 4.29ms | **2.40x** | **0.63ms** | 0.73ms | **1.16x** |
| lz4 | **1.63ms** | 2.65ms | **1.63x** | **0.25ms** | 0.29ms | **1.16x** |

#### 10M rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **138.6ms** | 178.3ms | **1.29x** | **10.95ms** | 35.93ms | **3.28x**\* |
| snappy | **221.3ms** | 288.9ms | **1.31x** | **39.35ms** | 46.60ms | **1.18x** |
| zstd | **204.7ms** | 396.4ms | **1.94x** | 65.84ms | **53.30ms** | 0.81x |
| lz4 | **204.5ms** | 304.2ms | **1.49x** | **24.84ms** | 39.50ms | **1.59x** |

#### 1M rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **8.85ms** | 18.04ms | **2.04x** | **0.04ms** | 2.63ms | 65.75x\* |
| snappy | **13.52ms** | 30.28ms | **2.24x** | 3.85ms | **3.64ms** | 0.95x |
| zstd | **16.43ms** | 39.87ms | **2.43x** | 6.12ms | **4.41ms** | 0.72x |
| lz4 | **13.77ms** | 31.18ms | **2.26x** | **2.40ms** | 3.34ms | **1.39x** |

#### 100K rows vs PyArrow

| Codec | Carquet Write | PyArrow Write | W ratio | Carquet Read | PyArrow Read | R ratio |
|-------|--------------|---------------|---------|-------------|--------------|---------|
| none | **1.38ms** | 2.06ms | **1.49x** | **0.02ms** | 0.24ms | 12x\* |
| snappy | **1.65ms** | 3.58ms | **2.17x** | **0.33ms** | 0.62ms | **1.88x** |
| zstd | **1.79ms** | 4.91ms | **2.74x** | **0.63ms** | 0.94ms | **1.49x** |
| lz4 | **1.63ms** | 3.21ms | **1.97x** | **0.25ms** | 0.41ms | **1.64x** |

\* Zero-copy mmap path

</details>

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
