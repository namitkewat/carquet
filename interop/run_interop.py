#!/usr/bin/env python3
"""
Carquet interoperability test suite.

Tests bidirectional compatibility with PyArrow, DuckDB, and fastparquet:
  - Read:  Carquet reads files written by other libraries
  - Write: Other libraries read files written by Carquet

Usage:
    python3 interop/run_interop.py                    # full suite
    python3 interop/run_interop.py --read-only         # carquet reads only
    python3 interop/run_interop.py --write-only        # roundtrip only
    python3 interop/run_interop.py -v                  # verbose output
    python3 interop/run_interop.py --keep-files        # keep temp files
    python3 interop/run_interop.py --build-dir ../build
"""

import argparse
import datetime
import json
import os
import platform
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

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

if not sys.stdout.isatty():
    BOLD = DIM = RST = RED = GRN = YLW = BLU = CYN = WHT = ""


def _bar(char="\u2500", width=62):
    return DIM + char * width + RST


# ── System info ──────────────────────────────────────────────────────────────

def get_cpu_name():
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


def get_system_info():
    return {
        "cpu": get_cpu_name(),
        "os": platform.platform(),
        "arch": platform.machine(),
    }


def get_carquet_version(build_dir):
    """Try to get carquet version from a built binary."""
    for binary in ["benchmark_carquet", "test_interop"]:
        path = os.path.join(build_dir, binary)
        if os.path.isfile(path):
            try:
                out = subprocess.check_output(
                    [path, "--version"], stderr=subprocess.DEVNULL, text=True,
                    timeout=5
                ).strip()
                if out:
                    return out
            except Exception:
                pass
    return "unknown"


# ── Library detection ────────────────────────────────────────────────────────

def detect_libraries():
    """Detect available Python libraries and their versions."""
    libs = {}
    for name in ["pyarrow", "duckdb", "fastparquet", "pandas", "numpy"]:
        try:
            mod = __import__(name)
            libs[name] = getattr(mod, "__version__", "?")
        except ImportError:
            pass
    return libs


# ── Read tests: Carquet reads files from other libraries ─────────────────────

def run_generate(script_dir, output_dir, verbose):
    """Generate test files using generate_test_files.py."""
    gen_script = script_dir / "generate_test_files.py"
    if not gen_script.exists():
        print(f"  {RED}Error:{RST} {gen_script} not found")
        return False

    proc = subprocess.run(
        [sys.executable, str(gen_script), str(output_dir)],
        capture_output=True, text=True, timeout=120
    )
    if proc.returncode != 0:
        print(f"  {RED}Error generating test files:{RST}")
        print(proc.stderr)
        return False

    if verbose:
        for line in proc.stdout.splitlines():
            print(f"    {DIM}{line}{RST}")

    # Count generated files
    count = sum(1 for f in output_dir.rglob("*.parquet"))
    print(f"  Generated {BOLD}{count}{RST} test files")
    return True


def run_read_tests(test_interop_bin, test_dir, verbose):
    """Run test_interop to read files from other libraries."""
    cmd = [str(test_interop_bin), "--dir", str(test_dir)]
    if verbose:
        cmd.append("-v")

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    results = {
        "tested": 0,
        "passed": 0,
        "failed": 0,
        "output": proc.stdout,
    }

    # Parse summary from test_interop output
    for line in proc.stdout.splitlines():
        if line.startswith("Files tested:"):
            try:
                results["tested"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("Passed:"):
            try:
                results["passed"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("Failed:"):
            try:
                results["failed"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass

    if verbose:
        for line in proc.stdout.splitlines():
            print(f"    {line}")

    return results


# ── Write tests: Other libraries read Carquet output ─────────────────────────

def run_roundtrip_writer(roundtrip_bin, output_dir):
    """Run roundtrip_writer to generate carquet files, return expected JSON."""
    proc = subprocess.run(
        [str(roundtrip_bin), str(output_dir)],
        capture_output=True, text=True, timeout=60
    )
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def verify_pyarrow(path, expected, file_info):
    """Verify a carquet-written file with PyArrow."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return ["pyarrow not available"]

    errors = []
    try:
        table = pq.read_table(path)
    except Exception as e:
        return [f"Failed to read: {e}"]

    if table.num_rows != expected["num_rows"]:
        errors.append(f"Row count: {table.num_rows} != {expected['num_rows']}")

    cols = file_info["columns"]

    # Check first values for each column
    for col_name, col_info in cols.items():
        try:
            actual = table.column(col_name).to_pylist()[:5]
            exp = col_info["first"]
            col_type = col_info.get("type", "")

            if col_type == "float":
                for i, (a, b) in enumerate(zip(actual, exp)):
                    if a is None and b is None:
                        continue
                    if a is None or b is None or abs(a - b) > 1e-4:
                        errors.append(f"{col_name}[{i}]: {a} != {b}")
                        break
            elif col_type == "double":
                for i, (a, b) in enumerate(zip(actual, exp)):
                    if a is None and b is None:
                        continue
                    if a is None or b is None or abs(a - b) > 1e-10:
                        errors.append(f"{col_name}[{i}]: {a} != {b}")
                        break
            elif col_type == "string":
                decoded = [
                    s.decode("utf-8") if isinstance(s, bytes) else s
                    for s in actual
                ]
                if decoded != exp:
                    errors.append(f"{col_name}: {decoded} != {exp}")
            else:
                if actual != exp:
                    errors.append(f"{col_name}: {actual} != {exp}")
        except Exception as e:
            errors.append(f"{col_name}: {e}")

    # Verify null counts
    verification = expected.get("verification", {})
    for key in ["null_count_string_col", "null_count_nullable_int"]:
        if key not in verification:
            continue
        col_name = key.replace("null_count_", "")
        try:
            actual_nulls = sum(
                1 for v in table.column(col_name).to_pylist() if v is None
            )
            if actual_nulls != verification[key]:
                errors.append(f"{col_name} nulls: {actual_nulls} != {verification[key]}")
        except Exception:
            pass

    # Verify aggregates
    if "int32_sum" in verification:
        try:
            actual_sum = sum(table.column("int32_col").to_pylist())
            if actual_sum != verification["int32_sum"]:
                errors.append(f"int32 sum: {actual_sum} != {verification['int32_sum']}")
        except Exception:
            pass

    return errors


def verify_duckdb(path, expected):
    """Verify a carquet-written file with DuckDB."""
    try:
        import duckdb
    except ImportError:
        return ["duckdb not available"]

    errors = []
    try:
        conn = duckdb.connect()
        df = conn.execute(f"SELECT * FROM read_parquet('{path}')").fetchdf()
        conn.close()
    except Exception as e:
        return [f"Failed to read: {e}"]

    if len(df) != expected["num_rows"]:
        errors.append(f"Row count: {len(df)} != {expected['num_rows']}")

    verification = expected.get("verification", {})
    if "int32_sum" in verification:
        try:
            actual_sum = int(df["int32_col"].sum())
            if actual_sum != verification["int32_sum"]:
                errors.append(f"int32 sum: {actual_sum} != {verification['int32_sum']}")
        except Exception:
            pass

    if "last_int32" in verification:
        try:
            actual_last = int(df["int32_col"].iloc[-1])
            if actual_last != verification["last_int32"]:
                errors.append(f"last int32: {actual_last} != {verification['last_int32']}")
        except Exception:
            pass

    return errors


def run_write_tests(roundtrip_bin, verbose):
    """Run roundtrip tests: carquet writes, other libraries verify."""
    results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        expected = run_roundtrip_writer(roundtrip_bin, tmpdir)
        if not expected:
            print(f"  {RED}Error:{RST} roundtrip_writer failed")
            return results

        num_files = len(expected.get("files", []))
        print(f"  Generated {BOLD}{num_files}{RST} files "
              f"({expected.get('num_rows', '?')} rows each)")

        for i, file_info in enumerate(expected.get("files", [])):
            path = file_info["path"]
            compression = file_info["compression"]
            tag = f"[{i + 1}/{num_files}]"

            entry = {
                "compression": compression,
                "pyarrow": None,
                "duckdb": None,
            }

            # PyArrow verification
            pa_errors = verify_pyarrow(path, expected, file_info)
            if pa_errors and pa_errors != ["pyarrow not available"]:
                entry["pyarrow"] = "FAIL"
                pa_status = f"{RED}FAIL{RST}"
                if verbose:
                    for e in pa_errors:
                        print(f"      {RED}{e}{RST}")
            elif pa_errors:
                entry["pyarrow"] = "SKIP"
                pa_status = f"{DIM}skip{RST}"
            else:
                entry["pyarrow"] = "PASS"
                pa_status = f"{GRN}OK{RST}"

            # DuckDB verification
            db_errors = verify_duckdb(path, expected)
            if db_errors and db_errors != ["duckdb not available"]:
                entry["duckdb"] = "FAIL"
                db_status = f"{RED}FAIL{RST}"
                if verbose:
                    for e in db_errors:
                        print(f"      {RED}{e}{RST}")
            elif db_errors:
                entry["duckdb"] = "SKIP"
                db_status = f"{DIM}skip{RST}"
            else:
                entry["duckdb"] = "PASS"
                db_status = f"{GRN}OK{RST}"

            print(f"  {CYN}{tag}{RST} {compression:<14} "
                  f"PyArrow {pa_status}  DuckDB {db_status}")

            results.append(entry)

    return results


# ── Summary ──────────────────────────────────────────────────────────────────

def print_read_summary(read_results):
    """Print read test summary."""
    tested = read_results["tested"]
    passed = read_results["passed"]
    failed = read_results["failed"]

    if failed > 0:
        color = RED
    elif passed == tested and tested > 0:
        color = GRN
    else:
        color = YLW

    print(f"  {BOLD}Read:{RST}   {color}{passed}/{tested} passed{RST}", end="")
    if failed > 0:
        print(f"  ({RED}{failed} failed{RST})", end="")
    print()


def print_write_summary(write_results):
    """Print write test summary table."""
    if not write_results:
        print(f"  {BOLD}Write:{RST}  {DIM}skipped{RST}")
        return

    pa_pass = sum(1 for r in write_results if r["pyarrow"] == "PASS")
    pa_total = sum(1 for r in write_results if r["pyarrow"] != "SKIP")
    db_pass = sum(1 for r in write_results if r["duckdb"] == "PASS")
    db_total = sum(1 for r in write_results if r["duckdb"] != "SKIP")

    pa_color = GRN if pa_pass == pa_total and pa_total > 0 else RED
    db_color = GRN if db_pass == db_total and db_total > 0 else RED

    parts = []
    if pa_total > 0:
        parts.append(f"PyArrow {pa_color}{pa_pass}/{pa_total}{RST}")
    if db_total > 0:
        parts.append(f"DuckDB {db_color}{db_pass}/{db_total}{RST}")

    print(f"  {BOLD}Write:{RST}  {', '.join(parts)}")

    # Detail table
    print()
    print(f"  {'Compression':<14} {'PyArrow':>8} {'DuckDB':>8}")
    print(f"  {_bar(width=32)}")

    for r in write_results:
        pa = r["pyarrow"] or "-"
        db = r["duckdb"] or "-"

        def color_status(s):
            if s == "PASS":
                return f"{GRN}PASS{RST}"
            elif s == "FAIL":
                return f"{RED}FAIL{RST}"
            return f"{DIM}{s}{RST}"

        print(f"  {r['compression']:<14} {color_status(pa):>17} {color_status(db):>17}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Carquet interoperability test suite"
    )
    parser.add_argument("--read-only", action="store_true",
                        help="Only test reading (carquet reads other libs' files)")
    parser.add_argument("--write-only", action="store_true",
                        help="Only test writing (other libs read carquet files)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep generated test files after run")
    parser.add_argument("--build-dir", default=None,
                        help="Path to CMake build directory (default: auto-detect)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    project_dir = script_dir.parent

    # Find build directory
    if args.build_dir:
        build_dir = Path(args.build_dir)
    else:
        build_dir = project_dir / "build"

    # Locate binaries
    test_interop_bin = build_dir / "test_interop"
    roundtrip_bin = build_dir / "roundtrip_writer"

    need_read = not args.write_only
    need_write = not args.read_only

    if need_read and not test_interop_bin.is_file():
        print(f"{RED}Error:{RST} {test_interop_bin} not found. Build first:")
        print(f"  cmake -B build -DCARQUET_BUILD_INTEROP=ON && cmake --build build")
        sys.exit(1)

    if need_write and not roundtrip_bin.is_file():
        print(f"{RED}Error:{RST} {roundtrip_bin} not found. Build first:")
        print(f"  cmake -B build -DCARQUET_BUILD_INTEROP=ON && cmake --build build")
        sys.exit(1)

    # Detect libraries
    libs = detect_libraries()
    sys_info = get_system_info()
    carquet_ver = get_carquet_version(str(build_dir))

    # ── Header ──
    print()
    print(f"  {BOLD}Carquet {carquet_ver} Interoperability Tests{RST}")
    print(f"  {_bar()}")
    print(f"  {DIM}CPU:{RST}  {sys_info['cpu']}")
    print(f"  {DIM}OS:{RST}   {sys_info['os']}")
    lib_strs = [f"{k} {v}" for k, v in libs.items()
                if k in ("pyarrow", "duckdb", "fastparquet")]
    if lib_strs:
        print(f"  {DIM}Libs:{RST} {', '.join(lib_strs)}")
    print(f"  {_bar()}")
    print()

    read_results = None
    write_results = None
    has_failures = False

    # ── Read phase ──
    if need_read:
        print(f"  {BOLD}Phase 1: Read Tests{RST} {DIM}(carquet reads other libs){RST}")
        print(f"  {_bar(width=42)}")

        # Generate test files
        if args.keep_files:
            test_dir = project_dir / "interop" / "test_files"
            test_dir.mkdir(exist_ok=True)
            cleanup_read = False
        else:
            test_dir = Path(tempfile.mkdtemp(prefix="carquet_interop_read_"))
            cleanup_read = True

        try:
            if not run_generate(script_dir, test_dir, args.verbose):
                has_failures = True
            else:
                read_results = run_read_tests(
                    test_interop_bin, test_dir, args.verbose
                )
                if read_results["failed"] > 0:
                    has_failures = True
        finally:
            if cleanup_read and test_dir.exists():
                shutil.rmtree(test_dir, ignore_errors=True)

        print()

    # ── Write phase ──
    if need_write:
        phase_num = "2" if need_read else "1"
        print(f"  {BOLD}Phase {phase_num}: Write Tests{RST} "
              f"{DIM}(other libs read carquet){RST}")
        print(f"  {_bar(width=42)}")

        write_results = run_write_tests(roundtrip_bin, args.verbose)
        write_failures = sum(
            1 for r in write_results
            if r["pyarrow"] == "FAIL" or r["duckdb"] == "FAIL"
        )
        if write_failures > 0:
            has_failures = True

        print()

    # ── Summary ──
    print(f"  {_bar('\u2550')}")
    print(f"  {BOLD}Summary{RST}")
    print(f"  {_bar('\u2550')}")
    print()

    if read_results:
        print_read_summary(read_results)

    if write_results is not None:
        print_write_summary(write_results)
    elif need_write:
        print(f"  {BOLD}Write:{RST}  {RED}failed to generate{RST}")

    print()

    if has_failures:
        print(f"  {RED}{BOLD}SOME TESTS FAILED{RST}")
    else:
        print(f"  {GRN}{BOLD}ALL TESTS PASSED{RST}")

    # ── JSON report ──
    report = {
        "carquet_version": carquet_ver,
        "timestamp": datetime.date.today().isoformat(),
        "system": sys_info,
        "libraries": libs,
    }
    if read_results:
        report["read"] = {
            "tested": read_results["tested"],
            "passed": read_results["passed"],
            "failed": read_results["failed"],
        }
    if write_results is not None:
        report["write"] = [
            {
                "compression": r["compression"],
                "pyarrow": r["pyarrow"],
                "duckdb": r["duckdb"],
            }
            for r in write_results
        ]

    date_str = datetime.date.today().strftime("%Y%m%d")
    os_tag = f"{platform.system().lower()}_{platform.machine()}"
    json_name = f"interop_{carquet_ver}_{os_tag}_{date_str}.json"
    json_path = script_dir / json_name
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  {DIM}Saved:{RST} {json_name}")
    print()

    sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    main()
