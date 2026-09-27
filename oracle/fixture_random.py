#!/usr/bin/env python3
# fixture_random.py: writes colmap-rust/tests/data/oracle/random.json, the libc++
# oracle for colmap_rust::math::random (Mt19937, the libc++ distributions and COLMAP's
# random.h API; Tier A, bit-exact). Read by colmap-rust/tests/math/rust_only_random_oracle.rs.
# Ported from colmap-sharp's oracle/fixture_random.py (same harness, same output).
#
# How it is produced: pycolmap 4.2.0 binds only set_random_seed, so no pycolmap call returns
# raw draws. COLMAP's RandomUniformInteger/Real/Gaussian are header templates over
# std::mt19937 and <random>, and the macOS wheel is built against libc++
# (oracle/probe_random.py). So this script compiles oracle/random_harness.cc with the
# system clang++ against libc++ (-stdlib=libc++, the macOS default), runs it, and records
# what it prints. Run it on macOS; on Linux, g++/libstdc++ would give different numbers
# and the script refuses.
#
# Floating-point contraction: Apple clang defaults to -ffp-contract=on, which may fuse
# a*b + c (uniform_real's (b-a)*u + a, normal's u*u + v*v and x*stddev + mean) into an FMA
# on arm64. colmap-rust never fuses (CLAUDE.md, "No FMA"). The harness is built twice,
# with the default and with -ffp-contract=off. "cases" holds the non-contracted numbers,
# which the Rust tests assert bit for bit. "contracted_cases" holds the default build's output
# for every case where it differs: the numbers the pycolmap wheel most likely produces, since
# its _core disassembles to tens of thousands of fmadd/fmsub instructions (it is built with
# contraction on). See docs/CPP_DIVERGENCES.md, entry 40.
#
# Usage: oracle/.venv/bin/python oracle/fixture_random.py   (numpy/pycolmap not needed)

import json
import pathlib
import platform
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "random_harness.cc"
OUTPUT = HERE.parent / "colmap-rust" / "tests" / "data" / "oracle" / "random.json"


def build_and_run(workdir, extra_flags):
    binary = workdir / ("harness" + "".join(f.replace("=", "_") for f in extra_flags))
    compile_args = ["clang++", "-std=c++17", "-O2", "-stdlib=libc++", *extra_flags,
                    str(SOURCE), "-o", str(binary)]
    subprocess.run(compile_args, check=True)
    return subprocess.run([str(binary)], capture_output=True, text=True, check=True).stdout


def parse(output):
    cases = {}
    for line in output.splitlines():
        name, kind, *values = line.split()
        if kind == "f":
            # Hex floats are exact; json writes Python floats with repr, which round-trips.
            parsed = [float.fromhex(v) for v in values]
        else:
            parsed = [int(v) for v in values]
        cases[name] = parsed
    return cases


def main():
    if platform.system() != "Darwin":
        sys.exit("run on macOS: the oracle is libc++'s <random>, which the pycolmap macOS wheel uses")

    compiler = subprocess.run(["clang++", "--version"], capture_output=True, text=True,
                              check=True).stdout.splitlines()[0]
    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        strict = build_and_run(workdir, ["-ffp-contract=off"])
        default = build_and_run(workdir, [])

    strict_cases = parse(strict)
    default_cases = parse(default)
    fixture = {
        "source": "oracle/fixture_random.py + oracle/random_harness.cc",
        "compiler": compiler,
        "platform": f"{platform.system()} {platform.machine()}",
        "flags": "-std=c++17 -O2 -stdlib=libc++ -ffp-contract=off",
        "cases": strict_cases,
        "contracted_cases": {name: values for name, values in default_cases.items()
                             if values != strict_cases[name]},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {len(fixture['cases'])} cases to {OUTPUT}")
    print(f"cases changed by default contraction: {sorted(fixture['contracted_cases'])}")


if __name__ == "__main__":
    main()
