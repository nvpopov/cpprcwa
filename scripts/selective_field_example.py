#!/usr/bin/env python3
"""Example: Solve_FieldFourierSelective / Solve_FieldOnGridSelective.

Demonstrates the "Selective" field-reconstruction methods, which return the
transmitted (forward-propagating) and/or reflected (backward-propagating)
contributions of the field independently.

The total field in a layer is always the sum of the two pieces:

    full = forward (transmitted) + backward (reflected)

`Solve_FieldFourier` / `Solve_FieldOnGrid` return the sum; the *Selective*
methods let you isolate either piece, e.g. to visualize only the reflected
diffraction orders in the incident medium, or only the transmitted field in
the exit medium.  The two kwargs are:

    include_forward  = True   # keep the forward-propagating  (transmitted) field
    include_backward = True   # keep the backward-propagating (reflected)  field

(both default to True, which reproduces the plain Solve_FieldFourier/OnGrid).

Usage:
    PYTHONPATH=build python3 scripts/selective_field_example.py [nG]
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


def e_density(e):
    """|E|^2 summed over the three components of one Fourier slice."""
    return float(np.sum(np.abs(e[0]) ** 2 + np.abs(e[1]) ** 2 + np.abs(e[2]) ** 2))


def main():
    nG = int(sys.argv[1]) if len(sys.argv) > 1 else 21

    # Square lattice of round holes (grcwa ex1 style), normal incidence.
    Nx, Ny = 40, 40
    obj = cpprcwa.obj(nG=nG, L1=[1.5, 0.0], L2=[0.0, 1.5],
                      freq=1.0, theta=0.0, phi=0.0)
    obj.Add_LayerUniform(1.0, 1.0 + 0j)   # incident vacuum (layer 0)
    obj.Add_LayerGrid(0.2, Nx, Ny)        # patterned layer (layer 1)
    obj.Add_LayerUniform(1.0, 1.0 + 0j)   # exit vacuum (layer 2)
    obj.Init_Setup()

    # A filled circle of index 12 in vacuum (strong scatterer).
    epgrid = np.ones((Nx, Ny), dtype=complex)
    u = np.linspace(0, 1, Nx)
    v = np.linspace(0, 1, Ny)
    for i in range(Nx):
        for j in range(Ny):
            dx, dy = u[i] - 0.5, v[j] - 0.5
            if dx * dx + dy * dy < 0.4 ** 2:
                epgrid[i, j] = 12.0 + 0j
    obj.GridLayer_geteps(epgrid.flatten())
    obj.MakeExcitationPlanewave(p_amp=1.0, p_phase=0.0, s_amp=0.0, s_phase=0.0)

    # ── Fourier-space: forward + backward == full ─────────────────────────────
    which, z = 0, 0.0                       # incident medium, front interface
    full = obj.Solve_FieldFourier(which, z)[0]      # [E, H]; each is [x, y, z]
    fwd = obj.Solve_FieldFourierSelective(which, z,
                                          include_forward=True,
                                          include_backward=False)[0]
    bwd = obj.Solve_FieldFourierSelective(which, z,
                                          include_forward=False,
                                          include_backward=True)[0]

    print("Solve_FieldFourierSelective — layer 0 (incident medium), z=0")
    for name, a, b, f in (("E", fwd[0], bwd[0], full[0]),
                          ("H", fwd[1], bwd[1], full[1])):
        for c, cn in enumerate("xyz"):
            d = np.max(np.abs(a[c] + b[c] - f[c]))
            print(f"  {name}{cn}: max |fwd + bwd - full| = {d:.3e}")

    print("\n  reflected |E|^2 (backward-only):", e_density(bwd[0]))
    print("  forward   |E|^2 (incident wave):", e_density(fwd[0]))
    print("  total     |E|^2 (full field)  :", e_density(full[0]))

    # ── Grid (real-space): forward + backward == full ─────────────────────────
    # Layer 0 is uniform, so the sampling grid must be given explicitly (the
    # plain/Selective grid methods throw ConfigError otherwise).
    Nxy = [Nx, Ny]
    gfull = obj.Solve_FieldOnGrid(which, z, Nxy)[0]
    gfwd = obj.Solve_FieldOnGridSelective(which, z, include_forward=True,
                                          include_backward=False, Nxy=Nxy)[0]
    gbwd = obj.Solve_FieldOnGridSelective(which, z, include_forward=False,
                                          include_backward=True, Nxy=Nxy)[0]

    print("\nSolve_FieldOnGridSelective — layer 0 (incident medium), z=0")
    print("  reflected |E|^2 grid shape:", gbwd[0][0].shape)
    d = np.max(np.abs(gfwd[0][0] + gbwd[0][0] - gfull[0][0]))
    print("  max |fwd + bwd - full| on Ex grid:", f"{d:.3e}")
    print("  reflected Ex map min/max:", f"{gbwd[0][0].real.min():.3f}",
          f"{gbwd[0][0].real.max():.3f}")

    # ── Exit medium: only the transmitted (forward) field is present ──────────
    last = obj.Layer_N - 1
    t = obj.Solve_FieldFourierSelective(last, 0.0, include_forward=True,
                                        include_backward=False)[0]
    r_last = obj.Solve_FieldFourierSelective(last, 0.0, include_forward=False,
                                             include_backward=True)[0]
    print(f"\nlayer {last} (exit medium), z=0")
    print("  transmitted |E|^2 (forward-only):", e_density(t[0]))
    print("  backward    |E|^2 (should be ~0):", e_density(r_last[0]))

    print("\nOK")


if __name__ == "__main__":
    main()
