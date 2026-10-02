from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aobasis")
except PackageNotFoundError:
    __version__ = "unknown"

from .base import BasisGenerator, ConcreteBasis, orthonormalize_modes, project_out, removal_basis
from .kl import KLBasisGenerator
from .zernike import ZernikeBasisGenerator
from .fourier import FourierBasisGenerator
from .zonal import ZonalBasisGenerator, ZonalFastBasisGenerator
from .hadamard import HadamardBasisGenerator
from .utils import (
    make_circular_actuator_grid,
    make_concentric_actuator_grid,
    plot_basis_modes,
    positions_from_mask,
)

__all__ = [
    "BasisGenerator",
    "ConcreteBasis",
    "orthonormalize_modes",
    "project_out",
    "removal_basis",
    "positions_from_mask",
    "KLBasisGenerator",
    "ZernikeBasisGenerator",
    "FourierBasisGenerator",
    "ZonalBasisGenerator",
    "ZonalFastBasisGenerator",
    "HadamardBasisGenerator",
    "make_circular_actuator_grid",
    "make_concentric_actuator_grid",
    "plot_basis_modes",
]
