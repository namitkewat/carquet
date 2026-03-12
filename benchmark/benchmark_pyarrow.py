#!/usr/bin/env python3
"""
Benchmark for PyArrow Parquet - compare with Carquet.

Methodology mirrors benchmark_carquet.c:
  - Write and read benchmarked separately
  - Page cache purged between write and read phases
  - 3 warmup + 11 measured iterations, trimmed median (drop min + max)
  - Data pre-generated outside timing loops
  - Arrow write/read options aligned to the Carquet benchmark where supported
"""

import os
import platform
import time
from functools import lru_cache

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

WARMUP_ITERATIONS = 3
BENCH_ITERATIONS = {"small": 51, "medium": 21, "large": 11, "xlarge": 11}
LCG_A = np.uint64(1103515245)
LCG_C = np.uint64(12345)
LCG_MASK = np.uint64(0xFFFFFFFF)
LCG_CHUNK_ROWS = 1_000_000
BOX_MULLER_PI = 3.14159265358979


def get_benchmark_zstd_level():
    raw = os.getenv("CARQUET_BENCH_ZSTD_LEVEL", "1").strip()
    try:
        level = int(raw)
    except ValueError:
        return 1
    return max(1, min(level, 22))


def trimmed_median(values):
    """Sort, drop min and max, return median of the rest."""
    s = sorted(values)
    n = len(s)
    if n <= 2:
        return s[n // 2]
    trimmed = s[1:-1]
    m = len(trimmed)
    if m % 2 == 1:
        return trimmed[m // 2]
    return (trimmed[m // 2 - 1] + trimmed[m // 2]) / 2


def purge_file_cache(filename):
    """Evict file from OS page cache so reads hit storage."""
    if platform.system() == "Darwin":
        import fcntl, shutil
        tmp = filename + ".nocache"
        # Copy through F_NOCACHE fd to avoid populating cache
        src = os.open(filename, os.O_RDONLY)
        dst = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        fcntl.fcntl(dst, fcntl.F_NOCACHE, 1)
        fcntl.fcntl(src, fcntl.F_NOCACHE, 1)
        while True:
            buf = os.read(src, 262144)
            if not buf:
                break
            os.write(dst, buf)
        os.close(src)
        os.close(dst)
        os.unlink(filename)
        os.rename(tmp, filename)
    elif platform.system() == "Linux":
        fd = os.open(filename, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def get_temp_dir():
    override = os.getenv("CARQUET_BENCH_TMPDIR", "").strip()
    if override:
        return override
    return "/tmp"


@lru_cache(maxsize=8)
def _lcg_advance_params(count):
    """Return affine params for advancing the C benchmark LCG by count steps."""
    idx = np.arange(count + 1, dtype=np.uint64)
    mul = np.ones(count + 1, dtype=np.uint64)
    add = np.zeros(count + 1, dtype=np.uint64)

    cur_mul = LCG_A
    cur_add = LCG_C
    bit = 1
    while bit <= count:
        mask = (idx & bit) != 0
        mul_mask = mul[mask]
        add_mask = add[mask]
        mul[mask] = (cur_mul * mul_mask) & LCG_MASK
        add[mask] = (cur_mul * add_mask + cur_add) & LCG_MASK
        cur_add = (cur_add * (cur_mul + np.uint64(1))) & LCG_MASK
        cur_mul = (cur_mul * cur_mul) & LCG_MASK
        bit <<= 1

    return mul[1:], add[1:]


def _lcg_rand_chunk(state, count):
    """Generate count outputs from the benchmark LCG, matching the C code."""
    mul, add = _lcg_advance_params(count)
    states = (mul * np.uint64(state) + add) & LCG_MASK
    values = ((states >> np.uint64(16)) & np.uint64(0x7FFF)).astype(np.uint32)
    return values, int(states[-1]) if count else state


def generate_data(num_rows):
    """Pre-generate test data using the same RNG sequence as benchmark_carquet.c."""
    ids = np.empty(num_rows, dtype=np.int64)
    values = np.empty(num_rows, dtype=np.float64)
    categories = np.empty(num_rows, dtype=np.int32)

    state = 42
    offset = 0
    while offset < num_rows:
        chunk_rows = min(LCG_CHUNK_ROWS, num_rows - offset)
        rand, state = _lcg_rand_chunk(state, chunk_rows * 4)
        rand = rand.reshape(chunk_rows, 4)

        ids[offset:offset + chunk_rows] = 1_000_000 + (rand[:, 0].astype(np.int64) % 9_000_000)
        u1 = (rand[:, 1].astype(np.float64) + 1.0) / 32768.0
        u2 = (rand[:, 2].astype(np.float64) + 1.0) / 32768.0
        normal = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * BOX_MULLER_PI * u2)
        values[offset:offset + chunk_rows] = np.abs(100.0 + 50.0 * normal)
        categories[offset:offset + chunk_rows] = (rand[:, 3] % 100).astype(np.int32)
        offset += chunk_rows

    return pa.table({"id": ids, "value": values, "category": categories})


def benchmark_write(filename, table, compression):
    """Write a Parquet file and return time_ms."""
    start = time.perf_counter()
    num_rows = len(table)
    rg_size = num_rows // 10 if num_rows > 1_000_000 else 100_000
    kwargs = {
        "compression": compression,
        "row_group_size": rg_size,
        "use_dictionary": False,
        "store_schema": False,
        "write_page_checksum": True,
    }
    if compression is not None:
        kwargs["use_byte_stream_split"] = ["value"]
    if compression == "zstd":
        kwargs["compression_level"] = get_benchmark_zstd_level()
    pq.write_table(table, filename, **kwargs)
    return (time.perf_counter() - start) * 1000


def benchmark_read(filename, expected_rows):
    """Read a Parquet file and return time_ms."""
    start = time.perf_counter()
    table = pq.read_table(
        filename,
        memory_map=True,
        use_threads=True,
        page_checksum_verification=False,
    )

    checksum = 0
    for col_name in table.column_names:
        arr = table[col_name].to_numpy()
        checksum += arr[0]

    elapsed_ms = (time.perf_counter() - start) * 1000

    if len(table) != expected_rows:
        raise ValueError(f"Row count mismatch: {len(table)} vs {expected_rows}")

    return elapsed_ms


def run_benchmark(name, num_rows, compression, compression_name):
    """Run a single benchmark configuration."""
    filename = os.path.join(
        get_temp_dir(),
        f"benchmark_{name}_{compression_name}_pyarrow.parquet",
    )

    print(f"\n=== {name} ({num_rows:,} rows, {compression_name}) ===")

    table = generate_data(num_rows)

    iters = BENCH_ITERATIONS.get(name, 21)

    # Warmup writes
    for _ in range(WARMUP_ITERATIONS):
        benchmark_write(filename, table, compression)

    # Benchmark writes
    write_times = []
    for _ in range(iters):
        write_times.append(benchmark_write(filename, table, compression))

    file_size = os.path.getsize(filename)

    # Purge cache once, then warmup reads (first read = cold, rest = warm)
    purge_file_cache(filename)

    for _ in range(WARMUP_ITERATIONS):
        benchmark_read(filename, num_rows)

    # Benchmark reads (warm cache — realistic for most workloads)
    read_times = []
    for _ in range(iters):
        read_times.append(benchmark_read(filename, num_rows))

    write_med = trimmed_median(write_times)
    read_med = trimmed_median(read_times)
    rows_per_sec_write = (num_rows / write_med) * 1000
    rows_per_sec_read = (num_rows / read_med) * 1000

    print(f"  Write: {write_med:.2f} ms ({rows_per_sec_write / 1e6:.2f} M rows/sec)")
    print(f"  Read:  {read_med:.2f} ms ({rows_per_sec_read / 1e6:.2f} M rows/sec)")
    print(f"  File:  {file_size / (1024 * 1024):.2f} MB ({file_size / num_rows:.2f} bytes/row)")

    print(f"CSV:pyarrow,{name},{compression_name},{num_rows},"
          f"{write_med:.2f},{read_med:.2f},{file_size}")

    os.remove(filename)


def main():
    print("PyArrow Benchmark")
    print(f"PyArrow version: {pa.__version__}")

    for name, rows in [("small", 100_000), ("medium", 1_000_000), ("large", 10_000_000),
                        ("xlarge", 100_000_000)]:
        for comp_name, comp in [("none", None), ("snappy", "snappy"), ("zstd", "zstd"), ("lz4", "lz4")]:
            run_benchmark(name, rows, comp, comp_name)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
