#!/usr/bin/env python3
"""Generate golden reference outputs from the grcwa Python library.

Runs the configured example through grcwa and writes reference outputs as
whitespace-delimited text files under tests/golden/ (readable by the C++
tests) plus optional .npy files for compare_results.py.

Usage:
    python3 scripts/generate_golden.py          # runs the S4 test config
    python3 scripts/generate_golden.py --example ex1
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "grcwa_orig"))
# Also try the sibling of the repo root (repo-adjacent grcwa_orig checkout).
_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)
for cand in (os.path.join(_repo, "..", "grcwa_orig"),
             os.path.join(os.path.dirname(_repo), "grcwa_orig")):
    if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "grcwa")):
        sys.path.insert(0, cand)
        break
import grcwa  # noqa: E402


def write_txt(path, arr):
    """Write a real or complex array as whitespace-delimited text.
    Complex arrays are written as 'real imag' pairs."""
    path = os.path.join(os.path.dirname(__file__), "..", "tests", "golden", path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if np.iscomplexobj(arr):
        arr = np.column_stack((arr.real, arr.imag))
    np.savetxt(path, arr.reshape(arr.shape[0], -1) if arr.ndim > 1 else arr[:, None])
    print(f"wrote {path} ({arr.shape})")


def s4_setup(planewave):
    """Build the S4 golden test object (mirrors tests/test_rcwa.py)."""
    nG = 101
    L1 = [0.1, 0]
    L2 = [0, 0.1]
    freq = 1.0
    theta = np.pi / 18
    phi = np.pi / 9
    Nx = Ny = 100
    epgrid = np.ones((Nx, Ny))
    x0 = np.linspace(0, 1, Nx)
    y0 = np.linspace(0, 1, Ny)
    x, y = np.meshgrid(x0, y0, indexing="ij")
    epgrid[(x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.16] = 12.0

    obj = grcwa.obj(nG, L1, L2, freq, theta, phi, verbose=0)
    obj.Add_LayerUniform(1.0, 1.0)
    obj.Add_LayerGrid(0.2, Nx, Ny)
    obj.Add_LayerUniform(1.0, 1.0)
    obj.Init_Setup()
    obj.GridLayer_geteps(epgrid.flatten())
    obj.MakeExcitationPlanewave(
        planewave["p_amp"], planewave["p_phase"],
        planewave["s_amp"], planewave["s_phase"], order=0)
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", default="s4",
                        choices=["s4", "ex1", "ex2", "ex4"])
    args = parser.parse_args()

    if args.example == "s4":
        for pol, pw, tag in [("p", {"p_amp": 1, "s_amp": 0, "p_phase": 0, "s_phase": 0}, "p"),
                             ("s", {"p_amp": 0, "s_amp": 1, "p_phase": 0, "s_phase": 0}, "s")]:
            obj = s4_setup(pw)
            R, T = obj.RT_Solve(normalize=0)
            write_txt(f"rt_s4_{tag}.txt", np.array([R, T]))
            write_txt(f"nG_s4_{tag}.txt", np.array([obj.nG]))
            print(f"s4 {pol}-pol: R={R:.12f} T={T:.12f} nG={obj.nG}")
    else:
        print(f"example runner for {args.example} not implemented yet; "
              "see examples/*.cpp which already embed the configs.")


if __name__ == "__main__":
    main()
