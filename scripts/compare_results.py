#!/usr/bin/env python3
"""Compare C++ outputs against grcwa golden references.

Reads whitespace-delimited golden files from tests/golden/ and compares them
against the corresponding C++-produced files (or prints instructions).

Usage:
    python3 scripts/compare_results.py rt_s4_p 0.85249901083265 0.8531730496
"""
import argparse
import os
import sys

import numpy as np


def load(path):
    """Load a whitespace-delimited golden file; auto-detects complex pairs."""
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == 2 and np.any(data[:, 1] != 0):
        return data[:, 0] + 1j * data[:, 1]
    return data[:, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("golden", help="golden file under tests/golden/")
    parser.add_argument("cpp_value", type=float, help="C++ scalar value")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    path = os.path.join(os.path.dirname(__file__), "..", "tests", "golden", args.golden)
    golden = load(path)
    if golden.size > 1:
        print(f"golden file has {golden.size} elements; comparing first to scalar")
    ref = np.real(golden[0]) if np.iscomplexobj(golden) else golden[0]
    rel = abs(args.cpp_value - ref) / max(abs(ref), 1e-30)
    status = "PASS" if rel < args.tolerance else "FAIL"
    print(f"{status}: cpp={args.cpp_value:.12f} golden={ref:.12f} rel={rel:.3e}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
