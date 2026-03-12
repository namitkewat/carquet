#!/usr/bin/env python3
"""
Carquet vs PyArrow benchmark orchestrator.

Runs each configuration independently with cooldown pauses between them
to avoid thermal throttling on fanless laptops (e.g. MacBook Air).

Usage:
    python3 benchmark/run_benchmark.py              # full benchmark
    python3 benchmark/run_benchmark.py --quick       # large/none only
    python3 benchmark/run_benchmark.py --cooldown 5  # 5s cooldown between configs
    python3 benchmark/run_benchmark.py --no-pyarrow  # carquet only
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import time

# ── ANSI helpers ─────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
RST = "\033[0m"
RED = "\033[31m"
GRN = "\033[32m"
YLW = "\033[33m"
BLU = "\033[34m"
CYN = "\033[36m"
WHT = "\033[37m"

# Disable colors if not a terminal
if not sys.stdout.isatty():
    BOLD = DIM = RST = RED = GRN = YLW = BLU = CYN = WHT = ""

def _bar(char="─", width=62):
    return DIM + char * width + RST


# ── System info ──────────────────────────────────────────────────────────────

def get_cpu_name():
    """Get CPU model string."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            if out:
                return out
        except Exception:
            pass
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def get_mem_gb():
    """Get total RAM in GB."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            return int(out) / (1024 ** 3)
        except Exception:
            pass
    return 0


def get_system_info():
    """Collect machine specs as a dict."""
    return {
        "cpu": get_cpu_name(),
        "ram_gb": round(get_mem_gb()),
        "os": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
    }


def get_carquet_version(binary):
    """Get carquet version from the benchmark binary."""
    try:
        out = subprocess.check_output(
            [binary, "--version"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        return out
    except Exception:
        return "unknown"


def print_system_info(info):
    print(f"  {DIM}CPU:{RST}  {info['cpu']}")
    if info["ram_gb"] > 0:
        print(f"  {DIM}RAM:{RST}  {info['ram_gb']} GB")
    print(f"  {DIM}OS:{RST}   {info['os']}")


# ── Benchmark runner ─────────────────────────────────────────────────────────

CONFIGS = [
    # (dataset, compression, rows) — largest first for best thermal conditions
    ("xlarge", "none",   100_000_000),
    ("xlarge", "snappy", 100_000_000),
    ("xlarge", "zstd",   100_000_000),
    ("large",  "none",    10_000_000),
    ("large",  "snappy",  10_000_000),
    ("large",  "zstd",    10_000_000),
    ("medium", "none",     1_000_000),
    ("medium", "snappy",   1_000_000),
    ("medium", "zstd",     1_000_000),
    ("small",  "none",       100_000),
    ("small",  "snappy",     100_000),
    ("small",  "zstd",       100_000),
]


def get_benchmark_zstd_level():
    raw = os.getenv("CARQUET_BENCH_ZSTD_LEVEL", "1").strip()
    try:
        level = int(raw)
    except ValueError:
        return 1
    return max(1, min(level, 22))


def parse_csv_line(line):
    """Parse a CSV: prefixed output line from benchmarks."""
    if not line.startswith("CSV:"):
        return None
    parts = line[4:].strip().split(",")
    return {
        "library":     parts[0],
        "dataset":     parts[1],
        "compression": parts[2],
        "rows":        int(parts[3]),
        "write_ms":    float(parts[4]),
        "read_ms":     float(parts[5]),
        "file_bytes":  int(parts[6]),
    }


def run_carquet_single(binary, dataset, compression):
    """Run carquet benchmark for a single config."""
    proc = subprocess.run(
        [binary, dataset, compression],
        capture_output=True, text=True, timeout=300
    )
    for line in proc.stdout.splitlines():
        r = parse_csv_line(line)
        if r:
            return r
    return None


def run_pyarrow_single(python, script, dataset, compression, rows):
    """Run pyarrow benchmark for a single config by invoking a helper snippet."""
    code = f"""
import sys, os
sys.path.insert(0, os.path.dirname({script!r}))
from benchmark_pyarrow import run_benchmark
comp_map = {{"none": None, "snappy": "snappy", "zstd": "zstd"}}
run_benchmark({dataset!r}, {rows}, comp_map[{compression!r}], {compression!r})
"""
    proc = subprocess.run(
        [python, "-c", code],
        capture_output=True, text=True, timeout=300
    )
    for line in proc.stdout.splitlines():
        r = parse_csv_line(line)
        if r:
            return r
    return None


def cooldown(seconds):
    """Visual cooldown timer."""
    if seconds <= 0:
        return
    for i in range(seconds, 0, -1):
        print(f"\r  {DIM}cooldown {i}s ...{RST}", end="", flush=True)
        time.sleep(1)
    print(f"\r{' ':40}\r", end="", flush=True)


# ── Result formatting ────────────────────────────────────────────────────────

def fmt_ms(ms):
    if ms is None:
        return "    -   "
    if ms >= 100:
        return f"{ms:>7.1f}ms"
    return f"{ms:>7.2f}ms"


def fmt_throughput(rows, ms):
    if ms is None or ms == 0:
        return ""
    mrows = (rows / ms) * 1000 / 1e6
    return f"{DIM}({mrows:>.0f}M r/s){RST}"


def fmt_speedup(carquet_ms, pyarrow_ms):
    if carquet_ms is None or pyarrow_ms is None or carquet_ms == 0:
        return "       "
    ratio = pyarrow_ms / carquet_ms
    if ratio >= 1.5:
        color = GRN
    elif ratio >= 1.0:
        color = WHT
    elif ratio >= 0.8:
        color = YLW
    else:
        color = RED
    return f"{color}{BOLD}{ratio:>5.2f}x{RST}"


def fmt_size(b):
    mb = b / (1024 * 1024)
    if mb >= 10:
        return f"{mb:>6.1f}MB"
    return f"{mb:>6.2f}MB"


def print_final_table(results, has_pyarrow):
    """Print the full comparison table."""
    print()
    print(f"  {BOLD}{'':15} {'Write':^24}{'':2}{'Read':^24}{'':2}{'Size':^16}{RST}")

    if has_pyarrow:
        print(f"  {'':15} {'Carquet':>8} {'PyArrow':>8} {'ratio':>6}"
              f"  {'Carquet':>8} {'PyArrow':>8} {'ratio':>6}"
              f"  {'Carquet':>7} {'PyArrow':>7}")
    else:
        print(f"  {'':15} {'time':>8} {'M r/s':>8}"
              f"  {'time':>8} {'M r/s':>8}"
              f"  {'bytes':>8}")
    print(f"  {_bar()}")

    # Group by dataset, print in display order (large → small)
    for dataset in ["xlarge", "large", "medium", "small"]:
        for compression in ["none", "snappy", "zstd"]:
            key = (dataset, compression)
            c = results.get(("carquet", key))
            p = results.get(("pyarrow", key))
            if not c and not p:
                continue

            label = f"{dataset}/{compression}"
            rows = c["rows"] if c else p["rows"]
            c_w = c["write_ms"] if c else None
            c_r = c["read_ms"] if c else None
            c_sz = c["file_bytes"] if c else None
            p_w = p["write_ms"] if p else None
            p_r = p["read_ms"] if p else None
            p_sz = p["file_bytes"] if p else None

            line = f"  {BOLD}{label:<15}{RST}"

            if has_pyarrow:
                # Write
                line += f" {fmt_ms(c_w)} {fmt_ms(p_w)} {fmt_speedup(c_w, p_w)}"
                # Read
                line += f"  {fmt_ms(c_r)} {fmt_ms(p_r)} {fmt_speedup(c_r, p_r)}"
                # Size
                line += f"  {fmt_size(c_sz) if c_sz else '':>7}"
                line += f" {fmt_size(p_sz) if p_sz else '':>7}"
            else:
                def throughput(r, ms):
                    if ms and ms > 0:
                        return f"{(r/ms)*1000/1e6:>7.1f}M"
                    return "      -"
                line += f" {fmt_ms(c_w)} {throughput(rows, c_w)}"
                line += f"  {fmt_ms(c_r)} {throughput(rows, c_r)}"
                line += f"  {fmt_size(c_sz) if c_sz else '':>8}"

            print(line)
        if dataset != "small":
            print()

    if has_pyarrow:
        print()
        print(f"  {DIM}ratio = PyArrow time / Carquet time (higher = Carquet faster){RST}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Carquet vs PyArrow benchmark with thermal management"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run only large/none config")
    parser.add_argument("--cooldown", type=int, default=3,
                        help="Seconds between configs (default: 3)")
    parser.add_argument("--no-pyarrow", action="store_true",
                        help="Skip PyArrow benchmarks")
    parser.add_argument("--no-carquet", action="store_true",
                        help="Skip Carquet benchmarks")
    args = parser.parse_args()

    # Locate binaries
    script_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(script_dir, "..", "build")
    carquet_bin = os.path.join(build_dir, "benchmark_carquet")

    if not args.no_carquet and not os.path.isfile(carquet_bin):
        print(f"{RED}Error:{RST} {carquet_bin} not found. Build first:")
        print(f"  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build")
        sys.exit(1)

    # Collect metadata
    sys_info = get_system_info()
    carquet_version = get_carquet_version(carquet_bin) if not args.no_carquet else "unknown"
    timestamp = datetime.date.today().isoformat()

    # Find Python with pyarrow
    python = sys.executable
    has_pyarrow = not args.no_pyarrow
    pyarrow_script = os.path.join(script_dir, "benchmark_pyarrow.py")
    pyarrow_version = None

    if has_pyarrow:
        try:
            out = subprocess.check_output(
                [python, "-c", "import pyarrow; print(pyarrow.__version__)"],
                stderr=subprocess.DEVNULL, text=True
            ).strip()
            pyarrow_version = out
        except Exception:
            print(f"{YLW}Warning:{RST} PyArrow not available, skipping")
            has_pyarrow = False

    # Select configs
    configs = CONFIGS
    if args.quick:
        configs = [c for c in configs if c[0] == "xlarge" and c[1] == "none"]

    # ── Header ──
    print()
    print(f"  {BOLD}Carquet {carquet_version} Benchmark{RST}")
    print(f"  {_bar()}")
    print_system_info(sys_info)
    if has_pyarrow:
        print(f"  {DIM}vs:{RST}   PyArrow {pyarrow_version}")
    zstd_level = get_benchmark_zstd_level()
    print(f"  {DIM}cfg:{RST}  {len(configs)} configs, {args.cooldown}s cooldown, "
          f"trimmed median, cache purged, 11-51 iters by size")
    print(f"  {DIM}zstd:{RST} level {zstd_level}")
    print(f"  {_bar()}")
    print()

    # ── Initial cooldown ──
    print(f"  {DIM}Letting system settle...{RST}")
    cooldown(args.cooldown)
    print()

    # ── Run benchmarks ──
    results = {}
    total = len(configs)

    for i, (dataset, compression, rows) in enumerate(configs):
        key = (dataset, compression)
        tag = f"[{i+1}/{total}]"
        label = f"{dataset}/{compression}"
        print(f"  {CYN}{tag}{RST} {BOLD}{label}{RST} {DIM}({rows:,} rows){RST}")

        carquet_r = None
        pyarrow_r = None

        # Carquet
        if not args.no_carquet:
            t0 = time.time()
            carquet_r = run_carquet_single(carquet_bin, dataset, compression)
            elapsed = time.time() - t0
            if carquet_r:
                results[("carquet", key)] = carquet_r
                mr = (rows / carquet_r["read_ms"]) * 1000 / 1e6
                sz = fmt_size(carquet_r["file_bytes"])
                print(f"       Carquet  W {fmt_ms(carquet_r['write_ms'])}  "
                      f"R {fmt_ms(carquet_r['read_ms'])} "
                      f"{DIM}({mr:.0f}M r/s, {sz}, {elapsed:.0f}s){RST}")
            else:
                print(f"       Carquet  {RED}failed{RST}")

        # Cooldown between libraries
        if not args.no_carquet and has_pyarrow:
            cooldown(max(1, args.cooldown // 2))

        # PyArrow
        if has_pyarrow:
            t0 = time.time()
            pyarrow_r = run_pyarrow_single(python, pyarrow_script,
                                           dataset, compression, rows)
            elapsed = time.time() - t0
            if pyarrow_r:
                results[("pyarrow", key)] = pyarrow_r
                mr = (rows / pyarrow_r["read_ms"]) * 1000 / 1e6
                sz = fmt_size(pyarrow_r["file_bytes"])
                print(f"       PyArrow  W {fmt_ms(pyarrow_r['write_ms'])}  "
                      f"R {fmt_ms(pyarrow_r['read_ms'])} "
                      f"{DIM}({mr:.0f}M r/s, {sz}, {elapsed:.0f}s){RST}")
            else:
                print(f"       PyArrow  {RED}failed{RST}")

        # Speedup summary for this config
        if carquet_r and pyarrow_r:
            ws = fmt_speedup(carquet_r["write_ms"], pyarrow_r["write_ms"])
            rs = fmt_speedup(carquet_r["read_ms"], pyarrow_r["read_ms"])
            print(f"       {DIM}speedup  W {ws}  R {rs}{RST}")

        # Cooldown between configs
        if i < total - 1:
            cooldown(args.cooldown)
        print()

    # ── Final table ──
    print(f"  {_bar('═')}")
    print(f"  {BOLD}Results{RST}")
    print(f"  {_bar('═')}")
    print_final_table(results, has_pyarrow)
    print()

    # ── Save JSON report ──
    report = {
        "carquet_version": carquet_version,
        "timestamp": timestamp,
        "system": sys_info,
        "config": {
            "warmup_iterations": 3,
            "bench_iterations": {"small": 51, "medium": 21, "large": 11, "xlarge": 11},
            "statistic": "trimmed_median",
            "cache_purge": True,
            "cooldown_seconds": args.cooldown,
        },
        "results": [],
    }
    if pyarrow_version:
        report["pyarrow_version"] = pyarrow_version

    for (lib, (ds, comp)), r in sorted(results.items()):
        report["results"].append({
            "library": lib,
            "dataset": ds,
            "compression": comp,
            "rows": r["rows"],
            "write_ms": round(r["write_ms"], 2),
            "read_ms": round(r["read_ms"], 2),
            "file_bytes": r["file_bytes"],
        })

    date_str = datetime.date.today().strftime("%Y%m%d")
    os_tag = f"{platform.system().lower()}_{platform.machine()}"
    json_name = f"bench_{carquet_version}_{os_tag}_{date_str}.json"
    json_path = os.path.join(script_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  {DIM}Saved:{RST} {json_name}")
    print()


if __name__ == "__main__":
    main()
