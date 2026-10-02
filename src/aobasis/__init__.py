from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("aobasis")
except PackageNotFoundError:
    __version__ = "unknown"

from .base import (
    BasisGenerator,
    ConcreteBasis,
    normalize_modes,
    orthonormalize_modes,
    project_out,
    removal_basis,
)
from .kl import KLBasisGenerator
from .dm_kl import DMKLBasisGenerator
from .zernike import ZernikeBasisGenerator
from .fourier import FourierBasisGenerator
from .zonal import ZonalBasisGenerator, ZonalFastBasisGenerator
from .hadamard import HadamardBasisGenerator
from .analysis import BasisReport, basis_report, command_to_mode_matrix, fit_coefficients
from .influence import fit_to_influence_functions, gaussian_influence_functions, make_pupil_points
from .utils import (
    make_circular_actuator_grid,
    make_concentric_actuator_grid,
    make_hexagonal_actuator_grid,
    plot_basis_modes,
    positions_from_mask,
)

__all__ = [
    "BasisGenerator",
    "ConcreteBasis",
    "normalize_modes",
    "orthonormalize_modes",
    "project_out",
    "removal_basis",
    "positions_from_mask",
    "KLBasisGenerator",
    "DMKLBasisGenerator",
    "ZernikeBasisGenerator",
    "FourierBasisGenerator",
    "ZonalBasisGenerator",
    "ZonalFastBasisGenerator",
    "HadamardBasisGenerator",
    "make_circular_actuator_grid",
    "make_concentric_actuator_grid",
    "make_hexagonal_actuator_grid",
    "make_pupil_points",
    "gaussian_influence_functions",
    "fit_to_influence_functions",
    "fit_coefficients",
    "command_to_mode_matrix",
    "basis_report",
    "BasisReport",
    "plot_basis_modes",
]
