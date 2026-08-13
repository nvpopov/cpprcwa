#!/usr/bin/env python3
"""Example: Solve_FieldFourier for several incident polarizations.

Shows how to set the incident polarization with MakeExcitationPlanewave and
recover the layer field with Solve_FieldFourier, for a few representative
polarization states, all against the SAME solver object.  Because the
S-matrix is excitation-independent and memoized, changing the polarization and
re-solving is cheap: the S-matrix is built once, and only the excitation →
field step re-runs for each polarization.

`p_amp` / `s_amp` are the physical p- and s-polarized plane-wave amplitudes and
`p_phase` / `s_phase` their phases; they are projected onto the (x, y)
transverse basis internally by MakeExcitationPlanewave.

Usage:
    PYTHONPATH=build python3 scripts/field_polarizations_example.py [nG]
"""
import os
import sys

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_here)

# Locate the cpprcwa nanobind module (build/, build-mkl/, or $CPPRCWA_BUILD).
for cand in (os.environ.get("CPPRCWA_BUILD"),
             os.path.join(_repo, "build"),
             os.path.join(_repo, "build-mkl")):
    if cand and os.path.isdir(cand):
        sys.path.insert(0, cand)
        break
import cpprcwa  # noqa: E402


def density(f):
    """Sum |·|^2 over the three components of one [x, y, z] Fourier slice."""
    return float(np.sum(np.abs(f[0]) ** 2 + np.abs(f[1]) ** 2 + np.abs(f[2]) ** 2))


def main():
    nG = int(sys.argv[1]) if len(sys.argv) > 1 else 21

    # Square lattice of round holes (ex1 style). Oblique incidence so the p and
    # s polarizations respond differently.
    Nx, Ny = 40, 40
    obj = cpprcwa.obj(nG=nG, L1=[1.5, 0.0], L2=[0.0, 1.5],
                      freq=1.0, theta=np.radians(30.0), phi=0.0)
    obj.Add_LayerUniform(1.0, 1.0 + 0j)   # incident vacuum (layer 0)
    obj.Add_LayerGrid(0.2, Nx, Ny)        # patterned layer (layer 1)
    obj.Add_LayerUniform(1.0, 1.0 + 0j)   # exit vacuum (layer 2)
    obj.Init_Setup()

    epgrid = np.ones((Nx, Ny), dtype=complex)
    u = np.linspace(0, 1, Nx)
    v = np.linspace(0, 1, Ny)
    for i in range(Nx):
        for j in range(Ny):
            dx, dy = u[i] - 0.5, v[j] - 0.5
            if dx * dx + dy * dy < 0.4 ** 2:
                epgrid[i, j] = 12.0 + 0j
    obj.GridLayer_geteps(epgrid.flatten())

    # (label, p_amp, p_phase, s_amp, s_phase) for several polarization states.
    pols = [
        ("p-pol",                1.0, 0.0,          0.0, 0.0),
        ("s-pol",                0.0, 0.0,          1.0, 0.0),
        ("+45 deg linear",       1.0, 0.0,          1.0, 0.0),
        ("-45 deg linear",       1.0, 0.0,          1.0, np.pi),
        ("circular (s +90 deg)", 1.0, 0.0,          1.0, np.pi / 2),
        ("circular (s -90 deg)", 1.0, 0.0,          1.0, -np.pi / 2),
    ]

    print(f"nG_req={nG} -> nG={obj.nG}   theta=30 deg, phi=0 deg")
    print(f"{'polarization':<20} {'|E|^2 layer 1':>15} {'|H|^2 layer 1':>15}")
    print("-" * 52)

    for label, p_amp, p_phase, s_amp, s_phase in pols:
        obj.MakeExcitationPlanewave(p_amp=p_amp, p_phase=p_phase,
                                    s_amp=s_amp, s_phase=s_phase)
        feh = obj.Solve_FieldFourier(1, 0.0)[0]   # [E, H] in the patterned layer
        e2 = density(feh[0])                      # feh[0] = [ex, ey, ez]
        h2 = density(feh[1])                      # feh[1] = [hx, hy, hz]
        print(f"{label:<20} {e2:15.8f} {h2:15.8f}")

    # The four "mixed" states share one |E|^2 value because for this symmetric
    # pattern the p- and s-scattered fields are orthogonal (so |E|^2 depends
    # only on |p_amp|^2 + |s_amp|^2). The complex fields themselves DO differ:
    # at phi=0 the s part maps onto Ey, so Ey[0] tracks s_amp * e^{i s_phase}.
    print("\nspecular (order-0) Ey value per state:")
    for label, p_amp, p_phase, s_amp, s_phase in pols:
        obj.MakeExcitationPlanewave(p_amp=p_amp, p_phase=p_phase,
                                    s_amp=s_amp, s_phase=s_phase)
        feh = obj.Solve_FieldFourier(1, 0.0)[0]
        print(f"  {label:<20} Ey[0] = {feh[0][1][0]:.6f}")

    # The excitation → field map is linear, so a +45° (p = s = 1) field is the
    # complex sum of the p-only and s-only fields. Verify this directly.
    def field_of(p_amp, s_amp):
        obj.MakeExcitationPlanewave(p_amp=p_amp, p_phase=0.0,
                                    s_amp=s_amp, s_phase=0.0)
        return np.array(obj.Solve_FieldFourier(1, 0.0)[0], dtype=complex)

    fp = field_of(1.0, 0.0)    # p only
    fs = field_of(0.0, 1.0)    # s only
    f45 = field_of(1.0, 1.0)   # +45 deg (p = s = 1)
    rel = np.abs(f45 - (fp + fs)).max() / max(np.abs(f45).max(), 1e-30)
    print(f"\nlinearity check: |(+45 deg) - (p + s)| / |+45 deg| = {rel:.3e}")

    print("OK")


if __name__ == "__main__":
    main()
