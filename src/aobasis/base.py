from abc import ABC, abstractmethod
import warnings
import numpy as np
from scipy.linalg import qr
from pathlib import Path
from typing import Callable, Optional, Sequence, Union
from .utils import plot_basis_modes


def _validate_positions_array(positions: np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(positions, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("positions must be a finite numeric array with shape (n_actuators, 2).") from exc

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("positions must have shape (n_actuators, 2).")
    if array.shape[0] == 0:
        raise ValueError("positions must contain at least one actuator.")
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

    def _removed_subspace(self, remove: "RemoveSpec" = None, ignore_piston: bool = False) -> np.ndarray:
        """Orthonormal ``(n_actuators, k)`` basis of the modes to keep out of the basis."""
        return removal_basis(self.positions, remove, ignore_piston=ignore_piston)

    def _take_outside(
        self,
        candidates: Callable[[int, int], np.ndarray],
        n_modes: int,
        removed: np.ndarray,
        n_available: Optional[int] = None,
    ) -> np.ndarray:
        """First ``n_modes`` candidates with ``removed`` projected out.

        ``candidates(start, count)`` returns candidate columns ``start`` to
        ``start + count - 1`` in order (fewer at the end of the supply).
        Candidates lying (numerically) inside the removed subspace, such as
        piston when piston is removed, are skipped.
        """
        if removed.shape[1] == 0:
            return candidates(0, n_modes)
        kept = []
        start = 0
        while sum(block.shape[1] for block in kept) < n_modes:
            need = n_modes - sum(block.shape[1] for block in kept)
            count = need + removed.shape[1]
            if n_available is not None:
                count = min(count, n_available - start)
            if count <= 0:
                break
            block = np.asarray(candidates(start, count), dtype=float)
            if block.shape[1] == 0:
                break
            start += block.shape[1]
            projected = block - removed @ (removed.T @ block)
            norms = np.linalg.norm(block, axis=0)
            inside = np.linalg.norm(projected, axis=0) <= _INSIDE_TOL * np.where(norms > 0, norms, 1.0)
            if inside.all():  # a whole block inside the removed modes: the supply is exhausted
                break
            kept.append(projected[:, ~inside][:, :need])
        modes = np.hstack(kept) if kept else np.zeros((self.n_actuators, 0))
        if modes.shape[1] < n_modes:
            raise ValueError(
                f"Only {modes.shape[1]} {self.__class__.__name__} modes lie outside the removed "
                f"modes on these {self.n_actuators} actuators; requested {n_modes}."
            )
        return modes

    def _finish(
        self, modes: np.ndarray, orthonormalize: bool = False, removed: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Store ``modes`` as float, optionally orthonormalized, warning if rank-deficient.

        ``removed`` (orthonormal columns) is projected out of the modes first.
        """
        modes = np.asarray(modes, dtype=float)
        if removed is not None and removed.shape[1]:
            modes = modes - removed @ (removed.T @ modes)
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

        Each generator takes its own keyword options (``ignore_piston``,
        ``orthonormalize``, ...) and rejects unknown ones with ``TypeError``.

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

        ``filepath`` may omit the ``.npz`` suffix that :meth:`save` (through
        ``np.savez``) adds.
        """
        filepath = Path(filepath)
        if not filepath.exists() and filepath.suffix != ".npz" and filepath.with_name(filepath.name + ".npz").exists():
            filepath = filepath.with_name(filepath.name + ".npz")
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
        instance.full_modes = modes
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


# A candidate whose residual outside the removed subspace is below this
# fraction of its norm is taken to lie inside it.
_INSIDE_TOL = 1e-8

RemoveSpec = Union[None, str, np.ndarray, Sequence[Union[str, np.ndarray]]]


def _named_modes(positions: np.ndarray, name: str) -> np.ndarray:
    x, y = positions[:, 0], positions[:, 1]
    columns = {
        "piston": [np.ones_like(x)],
        "tip": [x],
        "tilt": [y],
        "tiptilt": [x, y],
    }
    if name not in columns:
        raise ValueError(f"Unknown mode name {name!r}; expected one of {sorted(columns)}.")
    return np.column_stack(columns[name])


def removal_basis(positions: np.ndarray, remove: RemoveSpec = None, ignore_piston: bool = False) -> np.ndarray:
    """Orthonormal basis of the modes described by ``remove`` (and piston).

    Args:
        positions: ``(n_actuators, 2)`` actuator coordinates.
        remove: ``None``; a name (``"piston"``, ``"tip"`` (x), ``"tilt"`` (y)
            or ``"tiptilt"``); an array of shape ``(n_actuators,)`` or
            ``(n_actuators, k)``; or a list of names and arrays.
        ignore_piston: Include piston.

    Returns:
        ``(n_actuators, k)`` matrix with orthonormal columns spanning the
        given modes; linearly dependent inputs are merged (``k`` is the rank).
    """
    positions = np.asarray(positions, dtype=float)
    n_act = positions.shape[0]
    items = []
    if ignore_piston:
        items.append("piston")
    if remove is None:
        pass
    elif isinstance(remove, (str, np.ndarray)):
        items.append(remove)
    elif isinstance(remove, (list, tuple)):
        items.extend(remove)
    else:
        raise ValueError("remove must be a mode name, an array, or a list of names and arrays.")

    columns = []
    for item in items:
        if isinstance(item, str):
            columns.append(_named_modes(positions, item))
            continue
        if not isinstance(item, np.ndarray):
            raise ValueError("remove entries must be mode names or numpy arrays.")
        array = np.asarray(item, dtype=float)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[0] != n_act:
            raise ValueError(f"remove arrays must have shape ({n_act},) or ({n_act}, k), got {item.shape}.")
        if not np.all(np.isfinite(array)):
            raise ValueError("remove arrays must contain only finite values.")
        columns.append(array)

    if not columns:
        return np.zeros((n_act, 0))
    stacked = np.hstack(columns)
    q, r, _ = qr(stacked, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    if diag.size == 0 or diag[0] == 0:
        return np.zeros((n_act, 0))
    rank = int(np.count_nonzero(diag > diag[0] * max(stacked.shape) * 1e-12))
    return q[:, :rank]


def project_out(modes: np.ndarray, subspace: np.ndarray) -> np.ndarray:
    """Remove from each column of ``modes`` its component in the span of ``subspace``.

    ``subspace`` is ``(n_actuators,)`` or ``(n_actuators, k)``; its columns
    need not be orthonormal. The result is orthogonal to every column of
    ``subspace``.
    """
    modes = np.asarray(modes, dtype=float)
    if modes.ndim != 2:
        raise ValueError("modes must have shape (n_actuators, n_modes).")
    basis = removal_basis(np.zeros((modes.shape[0], 2)), np.asarray(subspace, dtype=float))
    return modes - basis @ (basis.T @ modes)


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
    """A basis loaded from disk (see :meth:`BasisGenerator.load`).

    ``full_modes`` holds every saved mode; :meth:`generate` returns the first
    ``n_modes`` of them and, like the other generators, stores them in
    ``modes``.
    """

    basis_type: Optional[str] = None
    full_modes: Optional[np.ndarray] = None

    def generate(self, n_modes: int) -> np.ndarray:
        if self.full_modes is None:
            if self.modes is None:
                raise NotImplementedError("This is a loaded basis container.")
            self.full_modes = self.modes
        n_modes = self._validate_n_modes(n_modes, max_modes=self.full_modes.shape[1])
        self.modes = self.full_modes[:, :n_modes]
        return self.modes
