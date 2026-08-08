"""Drop-in replacement for grcwa, backed by cpprcwa (nanobind).

Put the built `cpprcwa` extension module on the path (e.g. the CMake build
directory) and this package re-exports it under the `grcwa` name, so existing
`import grcwa` scripts run unchanged:

    PYTHONPATH=<build>:python python3 your_grcwa_script.py
"""
from cpprcwa import *  # noqa: F401,F403  (obj, Lattice_*, get_fft, get_ifft, Epsilon_fft)
from cpprcwa import obj  # noqa: F401

__version__ = "0.1.2"


def set_backend(name):
    """grcwa API compatibility: cpprcwa always uses the numpy backend."""
    if name not in ("numpy", "autograd"):
        raise ValueError(f"unknown backend '{name}'")


class _Backend:
    """Minimal stand-in for grcwa.backend (numpy)."""

    @staticmethod
    def pi():
        import numpy as np
        return np.pi


backend = _Backend()
