#!/usr/bin/env python3
"""Time grcwa (via the cpprcwa nanobind shim) on the EUV absorber geometry.

Same config as benchmarks/bench_euv_stack.cpp. Reports best-of-N setup and
rt_solve wall-clock for the specified (nG, quasi1d, theta, phi) combinations.

Usage:
  PYTHONPATH=<build-dir> python3 benchmarks/bench_grcwa.py [--quasi1d] [--theta D] [--phi D] [--runs N] nG1 nG2 ...
"""
import math
import sys
import time

import numpy as np
import grcwa

N_MO = 0.9226 + 0.0064j
N_SI = 0.9997 + 0.0018j
N_RU = 0.9114 + 0.0171j
N_TAN = 0.9562 + 0.0323j
LX, WX, LAMBDA = 3500.0, 500.0, 13.5
LY = 5.0 * LAMBDA
NX, NY, NB = 1000, 4, 40


def run_once(nG, quasi1d, theta_deg, phi_deg):
    obj = grcwa.obj(nG=nG, L1=[LX, 0.0], L2=[0.0, LY], freq=1.0 / LAMBDA + 0j,
                    theta=math.radians(theta_deg), phi=math.radians(phi_deg),
                    quasi1d=quasi1d)
    obj.Add_LayerUniform(1.0, 1.0)
    obj.Add_LayerGrid(60.0, NX, NY)
    obj.Add_LayerUniform(2.5, N_RU**2)
    for _ in range(NB):
        obj.Add_LayerUniform(2.8, N_MO**2)
        obj.Add_LayerUniform(4.2, N_SI**2)
    obj.Add_LayerUniform(1.0, N_SI**2)

    t0 = time.perf_counter()
    obj.Init_Setup()
    ep = np.ones((NX, NY), dtype=complex)
    bar = np.abs(np.linspace(0, 1, NX) - 0.5) < WX / (2.0 * LX)
    ep[bar, :] = N_TAN  # grid holds n, not n^2 (shared convention)
    obj.GridLayer_geteps(ep.flatten())
    t1 = time.perf_counter()
    obj.MakeExcitationPlanewave(p_amp=1.0, p_phase=0, s_amp=0, s_phase=0)
    R, T = obj.RT_Solve(normalize=1)
    t2 = time.perf_counter()
    return obj.nG, (t1 - t0) * 1e3, (t2 - t1) * 1e3, R


def main():
    args = sys.argv[1:]
    quasi1d, theta, phi, runs = False, 0.0, 0.0, 3
    sizes = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--quasi1d":
            quasi1d = True
        elif a == "--theta":
            theta = float(args[i + 1]); i += 1
        elif a == "--phi":
            phi = float(args[i + 1]); i += 1
        elif a == "--runs":
            runs = int(args[i + 1]); i += 1
        else:
            sizes.append(int(a))
        i += 1
    if not sizes:
        sizes = [201, 500]

    print(f"grcwa benchmark (quasi1d={quasi1d} theta={theta} phi={phi}, best of {runs})")
    print(f"{'nG':>6} {'2nG':>7} {'setup ms':>12} {'rt_solve ms':>12} {'total ms':>12} {'R':>9}")
    for nG in sizes:
        best = (1e300, 1e300)
        nG_out = R = 0
        for _ in range(runs):
            o, su, rt, R = run_once(nG, quasi1d, theta, phi)
            best = (min(best[0], su), min(best[1], rt))
            nG_out = o
        print(f"{nG_out:>6} {2 * nG_out:>7} {best[0]:>12.1f} {best[1]:>12.1f} "
              f"{best[0] + best[1]:>12.1f} {R:>9.6f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
