import numpy as np
from scipy.linalg import hadamard
from .base import BasisGenerator

class HadamardBasisGenerator(BasisGenerator):
    """
    Generates Hadamard modes.
    Useful for interaction matrix calibration (multiplexing).
    """
    
    def generate(
        self, n_modes: int, ignore_piston: bool = False, orthonormalize: bool = False, **kwargs
    ) -> np.ndarray:
        """
        Generate Hadamard modes (float entries of +1/-1).

        The Sylvester Hadamard matrix of the next power of two >= the number
        of actuators is truncated to the actuators (rows) and the first
        ``n_modes`` columns. The truncated columns are generally not
        orthogonal; pass ``orthonormalize=True`` to make them so (the entries
        are then no longer +/-1).

        Args:
            n_modes: Number of modes, at most the number of actuators.
            ignore_piston: Skip column 0, which is all ones.
            orthonormalize: Gram-Schmidt the modes in order so they are
                orthonormal on the actuator grid.
        """
        max_modes = self.n_actuators - (1 if ignore_piston else 0)
        n_modes = self._validate_n_modes(n_modes, max_modes=max_modes)
        if n_modes == 0:
            self.modes = np.zeros((self.n_actuators, 0), dtype=float)
            return self.modes

        size = 1
        while size < self.n_actuators:
            size *= 2

        first = 1 if ignore_piston else 0
        H = hadamard(size).astype(float)
        return self._finish(
            H[: self.n_actuators, first : first + n_modes], orthonormalize=orthonormalize
        )
