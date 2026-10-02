import numpy as np
from .base import BasisGenerator, RemoveSpec, removal_basis

class FourierBasisGenerator(BasisGenerator):
    """
    Generates Fourier modes (sine/cosine) on the actuator grid.
    """
    
    def __init__(self, positions: np.ndarray, pupil_diameter: float):
        super().__init__(positions)
        if not np.isscalar(pupil_diameter) or not np.isfinite(pupil_diameter) or pupil_diameter <= 0:
            raise ValueError("pupil_diameter must be a positive finite scalar.")
        self.pupil_diameter = pupil_diameter

    def generate(
        self,
        n_modes: int,
        ignore_piston: bool = False,
        orthonormalize: bool = False,
        remove: RemoveSpec = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate real Fourier modes: cos/sin pairs of increasing spatial frequency.

        Frequencies are integer multiples of one cycle per ``pupil_diameter``,
        taken in order of increasing ``|k|``. On a discrete actuator grid,
        frequencies above Nyquist alias onto lower ones, and some sine terms
        vanish on the grid. Any candidate that is (numerically) a combination
        of the modes already chosen is skipped, so the result has full rank.

        Args:
            n_modes: Number of modes, at most the number of actuators minus
                the number of removed modes.
            ignore_piston: Remove piston: leave out the constant mode and
                subtract the mean from every other mode (sampled sines and
                cosines are not zero-mean on a discrete grid).
            orthonormalize: Gram-Schmidt the modes in order so they are
                orthonormal on the actuator grid.
            remove: Further modes to project out, e.g. ``"tiptilt"`` (see
                :func:`aobasis.removal_basis`). Frequencies are chosen to be
                independent of them.

        Raises:
            ValueError: If the grid supports fewer than ``n_modes``
                independent Fourier modes.
        """
        removed = self._removed_subspace(remove, ignore_piston)
        n_modes = self._validate_n_modes(n_modes, max_modes=self.n_actuators - removed.shape[1])
        if n_modes == 0:
            self.modes = np.zeros((self.n_actuators, 0), dtype=float)
            return self.modes

        x = self.positions[:, 0]
        y = self.positions[:, 1]
        f0 = 1.0 / self.pupil_diameter

        # q holds an orthonormal basis of the span so far: the removed modes
        # and piston first (piston even when it is removed, so the chosen
        # modes are independent of it), then each chosen mode.
        # Candidates have unit amplitude, so residuals are compared with the
        # norm of a unit-amplitude mode: sine terms that vanish on the grid
        # have a tiny norm and a relative test would accept them.
        threshold = self._DEPENDENCE_TOL * np.sqrt(self.n_actuators)
        seed = removal_basis(self.positions, removed, ignore_piston=True)
        q = np.empty((self.n_actuators, n_modes + seed.shape[1]))
        q[:, : seed.shape[1]] = seed
        count = seed.shape[1]
        piston = np.ones_like(x)
        piston_removed = np.linalg.norm(piston - removed @ (removed.T @ piston)) <= 1e-8 * np.sqrt(
            self.n_actuators
        )
        chosen = [] if piston_removed else [piston]

        def add_batch(candidates: list) -> None:
            nonlocal count
            block = np.column_stack(candidates)
            residual = block.copy()
            for _ in range(2):  # classical Gram-Schmidt twice ("twice is enough")
                residual -= q[:, :count] @ (q[:, :count].T @ residual)
            batch_start = count
            for i in range(block.shape[1]):
                if len(chosen) >= n_modes:
                    return
                r = residual[:, i]
                new = q[:, batch_start:count]
                for _ in range(2):
                    r = r - new @ (new.T @ r)
                norm = np.linalg.norm(r)
                if norm > threshold:
                    q[:, count] = r / norm
                    count += 1
                    chosen.append(block[:, i])

        # Integer k-vectors in half the plane ((k) and (-k) give the same
        # real modes), taken in order of |k|. Regular grids fill up by
        # Nyquist; irregular ones may need higher frequencies, so the pool
        # grows (up to 4x) until enough modes are found.
        k_max = int(np.ceil(np.sqrt(self.n_actuators))) + 2
        k_limit = 4 * k_max
        tried = set()
        while len(chosen) < n_modes and k_max <= k_limit:
            k_pairs = [
                (kx, ky)
                for kx in range(0, k_max + 1)
                for ky in range(-k_max, k_max + 1)
                if (kx > 0 or ky > 0) and (kx, ky) not in tried
            ]
            k_pairs.sort(key=lambda k: (k[0] ** 2 + k[1] ** 2, -k[0], -k[1]))
            for start in range(0, len(k_pairs), self._BATCH):
                if len(chosen) >= n_modes:
                    break
                batch = []
                for kx, ky in k_pairs[start : start + self._BATCH]:
                    tried.add((kx, ky))
                    arg = 2 * np.pi * f0 * (kx * x + ky * y)
                    batch.extend((np.cos(arg), np.sin(arg)))
                add_batch(batch)
            k_max *= 2

        if len(chosen) < n_modes:
            raise ValueError(
                f"Only {len(chosen)} independent Fourier modes (frequencies in steps "
                f"of 1/pupil_diameter) exist on these {self.n_actuators} actuators; "
                f"requested {n_modes}."
            )
        return self._finish(np.column_stack(chosen), orthonormalize=orthonormalize, removed=removed)

    _BATCH = 32
    _DEPENDENCE_TOL = 1e-6
