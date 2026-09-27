#!/usr/bin/env python3
# probe_random.py: which C++ standard library's <random> distributions does the pinned
# pycolmap wheel run? The answer decides how colmap_rust::math::random implements
# std::uniform_int_distribution / uniform_real_distribution / normal_distribution (their
# algorithms are implementation-defined: libc++ and libstdc++ give different numbers from
# the same mt19937 stream). See CLAUDE.md's PRNG rule.
#
# Method:
# 1. Look for bindings that expose colmap's random functions directly. pycolmap 4.2.0 has
#    only set_random_seed; RandomUniformInteger/Real/Gaussian are not bound.
# 2. So decide it from the binary: the distributions are header-only templates inlined
#    into _core, so they come from whichever standard library the extension was compiled
#    against. libc++ mangles std:: as std::__1 (`NSt3__1`) and ships as libc++.1.dylib /
#    libc++.so.1; libstdc++ uses std::__cxx11 and GLIBCXX_* symbol versions and links
#    libstdc++.so.6.
#
# Usage: oracle/.venv/bin/python oracle/probe_random.py
# Writes nothing; prints evidence and a verdict.

import pathlib
import platform
import re
import subprocess
import sys

import pycolmap


def run(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        return f"<{' '.join(args)} failed: {error}>"


def main():
    print(f"pycolmap {pycolmap.__version__} on {platform.system()} {platform.machine()}")

    exposed = sorted(n for n in dir(pycolmap) if re.search(r"random|seed|gaussian", n, re.I))
    print(f"random-related bindings: {exposed}")

    package_dir = pathlib.Path(pycolmap.__file__).parent
    cores = sorted(package_dir.glob("_core*.so")) + sorted(package_dir.glob("_core*.pyd"))
    if not cores:
        sys.exit(f"no _core extension found in {package_dir}")
    core = cores[0]
    print(f"extension: {core}")

    if platform.system() == "Darwin":
        linked = run(["otool", "-L", str(core)])
        undefined = run(["nm", "-u", str(core)])
    else:
        linked = run(["ldd", str(core)])
        undefined = run(["nm", "-D", "-u", str(core)])

    libcxx_linked = bool(re.search(r"libc\+\+(\.1)?\.(dylib|so)", linked))
    libstdcxx_linked = "libstdc++" in linked
    libcxx_symbols = len(re.findall(r"NSt3__1|St3__1", undefined))
    libstdcxx_symbols = len(re.findall(r"cxx11|GLIBCXX", undefined))

    print(f"links libc++: {libcxx_linked}; links libstdc++: {libstdcxx_linked}")
    print(f"undefined std::__1 (libc++) symbols: {libcxx_symbols}")
    print(f"undefined __cxx11/GLIBCXX (libstdc++) symbols: {libstdcxx_symbols}")

    if libcxx_linked and libcxx_symbols > 0 and not libstdcxx_linked and libstdcxx_symbols == 0:
        print("VERDICT: libc++ (LLVM) distributions")
    elif libstdcxx_linked and libstdcxx_symbols > 0 and not libcxx_linked:
        print("VERDICT: libstdc++ (GNU) distributions")
    else:
        print("VERDICT: inconclusive, inspect the evidence above")


if __name__ == "__main__":
    main()
