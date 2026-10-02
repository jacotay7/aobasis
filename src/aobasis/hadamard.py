import numpy as np
from typing import Optional
from scipy.linalg import hadamard
from .base import BasisGenerator, RemoveSpec, _check_normalize

class HadamardBasisGenerator(BasisGenerator):
    """
    Generates Hadamard modes.
    Useful for interaction matrix calibration (multiplexing).
    """
    
    def generate(
        self,
        n_modes: int,
        ignore_piston: bool = False,
        orthonormalize: bool = False,
        remove: RemoveSpec = None,
        normalize: Optional[str] = None,
    ) -> np.ndarray:
        """
        Generate Hadamard modes (float entries of +1/-1).

        The Sylvester Hadamard matrix of the next power of two >= the number
        of actuators is truncated to the actuators (rows) and the first
        ``n_modes`` columns. The truncated columns are generally not
        orthogonal; pass ``orthonormalize=True`` to make them so (the entries
        are then no longer +/-1).

        Args:
            n_modes: Number of modes, at most the number of actuators minus
                the number of removed modes.
            ignore_piston: Remove piston: column 0 (all ones) is skipped and
                the mean is subtracted from the other columns, which are not
                zero-mean once truncated. Entries are then no longer +/-1.
            orthonormalize: Gram-Schmidt the modes in order so they are
                orthonormal on the actuator grid.
            remove: Further modes to project out (see
                :func:`aobasis.removal_basis`).
            normalize: Scale each mode to unit ``"rms"``, ``"l2"``,
                ``"peak"`` or ``"pv"`` after everything else (see
                :func:`aobasis.normalize_modes`). ``None`` keeps the
                generator's own scale.
        """
        _check_normalize(normalize)
        removed = self._removed_subspace(remove, ignore_piston)
        n_modes = self._validate_n_modes(n_modes, max_modes=self.n_actuators - removed.shape[1])
        if n_modes == 0:
            self.modes = np.zeros((self.n_actuators, 0), dtype=float)
            return self.modes

        size = 1
        while size < self.n_actuators:
            size *= 2
        H = hadamard(size)[: self.n_actuators].astype(float)

        modes = self._take_outside(
            lambda start, count: H[:, start : start + count], n_modes, removed, n_available=size
        )
        return self._finish(modes, orthonormalize=orthonormalize, removed=removed, normalize=normalize)
