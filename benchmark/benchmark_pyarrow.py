#!/usr/bin/env python3
"""
Benchmark for PyArrow Parquet - compare with Carquet

Methodology mirrors benchmark_carquet.c:
  - Write and read benchmarked separately
  - Page cache purged between write and read phases
  - 3 warmup + 11 measured iterations, trimmed median (drop min + max)
  - Data pre-generated outside timing loops
"""

import os
import platform
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

WARMUP_ITERATIONS = 3
BENCH_ITERATIONS = {"small": 51, "medium": 21, "large": 11, "xlarge": 11}


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


def generate_data(num_rows):
    """Pre-generate test data (outside timing loop)."""
    np.random.seed(42)
    ids = np.random.randint(1_000_000, 9_999_999, size=num_rows, dtype=np.int64)
    values = np.abs(np.random.normal(100.0, 50.0, size=num_rows))
    categories = np.random.randint(0, 100, size=num_rows, dtype=np.int32)
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
    }
    if compression == "zstd":
        kwargs["compression_level"] = get_benchmark_zstd_level()
    pq.write_table(table, filename, **kwargs)
    return (time.perf_counter() - start) * 1000


def benchmark_read(filename, expected_rows):
    """Read a Parquet file and return time_ms."""
    start = time.perf_counter()
    table = pq.read_table(filename)

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
    filename = f"/tmp/benchmark_{name}_{compression_name}_pyarrow.parquet"

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
        for comp_name, comp in [("none", None), ("snappy", "snappy"), ("zstd", "zstd")]:
            run_benchmark(name, rows, comp, comp_name)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()
