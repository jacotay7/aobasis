from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy.linalg import qr
from pathlib import Path
from typing import Tuple, Optional, Union
from .utils import plot_basis_modes


def _validate_positions_array(positions: np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(positions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("positions must be a finite numeric array with shape (n_actuators, 2).") from exc

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("positions must have shape (n_actuators, 2).")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions must contain only finite values.")

    return array

class BasisGenerator(ABC):
    """
    Abstract base class for AO basis generators.
    """
    
    def __init__(self, positions: np.ndarray):
        """
        Args:
            positions: (N, 2) array of actuator coordinates (x, y) in meters.
        """
        self.positions = _validate_positions_array(positions)
        self.n_actuators = self.positions.shape[0]
        self.modes: Optional[np.ndarray] = None

    def _validate_n_modes(self, n_modes: int, max_modes: Optional[int] = None) -> int:
        if isinstance(n_modes, bool) or not isinstance(n_modes, (int, np.integer)):
            raise ValueError("n_modes must be an integer.")

        n_modes = int(n_modes)
        if n_modes < 0:
            raise ValueError("n_modes must be non-negative.")
        if max_modes is not None and n_modes > max_modes:
            raise ValueError(f"Cannot generate {n_modes} modes; maximum available is {max_modes}.")

        return n_modes

    def _finish(self, modes: np.ndarray, orthonormalize: bool = False) -> np.ndarray:
        """Store ``modes`` as float, optionally orthonormalized, warning if rank-deficient."""
        modes = np.asarray(modes, dtype=float)
        n_modes = modes.shape[1]
        if n_modes:
            rank = _numerical_rank(modes)
            if rank < n_modes:
                warnings.warn(
                    f"{self.__class__.__name__}: {n_modes} modes have rank {rank} on "
                    f"these {self.n_actuators} actuators; some modes are linearly dependent.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            if orthonormalize:
                modes = orthonormalize_modes(modes)
        self.modes = modes
        return modes

    @abstractmethod
    def generate(self, n_modes: int, **kwargs) -> np.ndarray:
        """
        Generate the basis modes.
        
        Args:
            n_modes: Number of modes to generate.
            
        Returns:
            modes: (n_actuators, n_modes) matrix.
        """
        pass
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the generated basis and actuator positions to a .npz file.
        """
        if self.modes is None:
            raise ValueError("No modes generated yet. Call generate() first.")
            
        np.savez(
            filepath,
            modes=self.modes,
            positions=self.positions,
            basis_type=getattr(self, 'basis_type', None) or self.__class__.__name__,
        )
        
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BasisGenerator':
        """
        Load a basis saved with :meth:`save`.

        Returns a :class:`ConcreteBasis` holding the saved modes and positions;
        its ``basis_type`` attribute is the name of the generator that made it.
        Generator parameters (pupil size, r0, ...) are not saved, so the
        original generator is not rebuilt. Calling ``load`` on a specific
        generator class (e.g. ``KLBasisGenerator.load``) raises ``ValueError``
        if the file was saved by a different generator.
        """
        with np.load(filepath) as data:
            positions = data['positions']
            modes = data['modes']
            basis_type = str(data['basis_type']) if 'basis_type' in data else None

        if (
            basis_type is not None
            and cls not in (BasisGenerator, ConcreteBasis)
            and basis_type != cls.__name__
        ):
            raise ValueError(f"{filepath} holds a {basis_type} basis, not {cls.__name__}.")

        instance = ConcreteBasis(positions)
        instance.modes = modes
        instance.basis_type = basis_type
        return instance

    def plot(self, count: int = 6, outfile: Optional[Union[str, Path]] = None, **kwargs):
        """Plot the generated modes."""
        if self.modes is None:
            raise ValueError("No modes to plot.")
        plot_basis_modes(self.modes, self.positions, count=count, outfile=outfile, **kwargs)

def _numerical_rank(modes: np.ndarray) -> int:
    """Rank from a column-pivoted QR, with ``np.linalg.matrix_rank``'s tolerance.

    Several times cheaper than the SVD that ``matrix_rank`` uses, and
    rank-revealing in practice.
    """
    r = qr(modes, mode="r", pivoting=True, check_finite=False)[0]
    diag = np.abs(np.diag(r))
    if diag.size == 0 or diag[0] == 0:
        return 0
    tol = diag[0] * max(modes.shape) * np.finfo(float).eps
    return int(np.count_nonzero(diag > tol))


def orthonormalize_modes(modes: np.ndarray) -> np.ndarray:
    """Gram-Schmidt the columns of ``modes`` in order (QR), keeping each sign.

    Column ``k`` of the result spans the same space as columns ``0..k`` of the
    input, so the modal ordering is preserved. Columns are unit L2 norm.
    """
    modes = np.asarray(modes, dtype=float)
    if modes.shape[1] == 0:
        return modes
    q, r = np.linalg.qr(modes)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs


class ConcreteBasis(BasisGenerator):
    """Helper class for loading existing bases."""

    basis_type: Optional[str] = None

    def generate(self, n_modes: int, **kwargs) -> np.ndarray:
        if self.modes is None:
            raise NotImplementedError("This is a loaded basis container.")
        n_modes = self._validate_n_modes(n_modes, max_modes=self.modes.shape[1])
        return self.modes[:, :n_modes]
