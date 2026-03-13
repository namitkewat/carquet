# Carquet Fuzzing

Fuzz targets for testing carquet with random and malformed inputs using libFuzzer or AFL++.

## Quick Start

```bash
# Build and run the reader fuzzer for 5 minutes
python3 fuzz/run_fuzzer.py reader

# Run all fuzzers (5 min each)
python3 fuzz/run_fuzzer.py all

# Run for 1 hour with 4 parallel jobs
python3 fuzz/run_fuzzer.py all --time 3600 --jobs 4

# Just build
python3 fuzz/run_fuzzer.py build

# List targets and corpus status
python3 fuzz/run_fuzzer.py list
```

## Requirements

- **Clang** with libFuzzer support (Clang 6.0+)
- On macOS: `brew install llvm`
- On Linux: `apt install clang`

## Fuzz Targets

| Target | Modes | Coverage |
|--------|-------|----------|
| `fuzz_reader` | 1 | Full file reader: buffer open, schema, batch reader, column reader, type names |
| `fuzz_writer` | 1 | Schema creation, all physical types (incl. BYTE_ARRAY), all codecs, config variants, write-read roundtrip |
| `fuzz_compression` | 3 | Snappy/LZ4/GZIP/ZSTD: decompress malformed, compress-decompress roundtrip, undersized buffer |
| `fuzz_encodings` | 17 | RLE, Delta INT32/INT64, Plain (bool/int32/int64/float/double), Dictionary (int32/int64/float/double), BSS (float/double/generic), Delta Length Byte Array, RLE levels |
| `fuzz_thrift` | 6 | Primitives, struct parsing, containers (list/map), file metadata, page headers, encoder-decoder roundtrip |
| `fuzz_roundtrip` | 11 | Delta INT32/INT64, LZ4/Snappy/GZIP/ZSTD, BSS float/double, RLE, Dictionary INT32, Plain INT32 |

## Commands

```bash
# Fuzz a specific target
python3 fuzz/run_fuzzer.py <target> [--time N] [--jobs N]

# Run all targets
python3 fuzz/run_fuzzer.py all [--time N] [--jobs N]

# Build only
python3 fuzz/run_fuzzer.py build [--no-sanitizers]

# List targets and corpus info
python3 fuzz/run_fuzzer.py list

# Minimize a corpus
python3 fuzz/run_fuzzer.py minimize <target>

# Generate coverage report
python3 fuzz/run_fuzzer.py coverage <target>
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--time N` | 300 | Seconds per target |
| `--jobs N` | 1 | Parallel fuzzing jobs |
| `--no-sanitizers` | off | Disable ASan/UBSan |
| `--build-dir DIR` | `build-fuzz` | Build directory |

Extra libFuzzer flags can be passed after `--`:
```bash
python3 fuzz/run_fuzzer.py reader --time 600 -- -max_len=4096 -only_ascii=1
```

## Building Manually

### libFuzzer (recommended)

```bash
cmake -B build-fuzz \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
    -DCARQUET_BUILD_FUZZ=ON \
    -DCARQUET_BUILD_TESTS=OFF

cmake --build build-fuzz
```

### AFL++

```bash
CC=afl-clang-fast cmake -B build-afl \
    -DCARQUET_BUILD_FUZZ=ON \
    -DCARQUET_FUZZ_ENGINE=AFL \
    -DCMAKE_BUILD_TYPE=Debug

cmake --build build-afl
afl-fuzz -i corpus_reader -o findings -- ./build-afl/fuzz/fuzz_reader_afl @@
```

## Crash Analysis

When a crash is found:

1. Artifacts are saved to `build-fuzz/fuzz/artifacts_<target>/`
2. Reproduce: `./build-fuzz/fuzz/fuzz_<target> <artifact>`
3. Stack trace: `ASAN_OPTIONS=symbolize=1 ./build-fuzz/fuzz/fuzz_<target> <artifact>`
