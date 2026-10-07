import numpy as np
from typing import Optional
from scipy.linalg import hadamard, qr
from .base import BasisGenerator, RemoveSpec, _check_normalize, removal_basis

CONSTRUCTIONS = ("sylvester", "smallest")
SELECTIONS = ("first", "balanced")


def _is_prime(q: int) -> bool:
    if q < 2:
        return False
    if q % 2 == 0:
        return q == 2
    return all(q % d for d in range(3, int(q**0.5) + 1, 2))


def _quadratic_character(q: int) -> np.ndarray:
    """chi(d) for d = 0..q-1 modulo the prime q: 0, +1 (residue) or -1."""
    chi = -np.ones(q, dtype=int)
    chi[(np.arange(1, q, dtype=np.int64) ** 2) % q] = 1
    chi[0] = 0
    return chi


def _paley(q: int, kind: int) -> np.ndarray:
    """Paley I (order q + 1, q = 3 mod 4) or II (order 2 (q + 1), q = 1 mod 4) Hadamard matrix."""
    chi = _quadratic_character(q)
    idx = np.arange(q)
    jacobsthal = chi[(idx[None, :] - idx[:, None]) % q]
    if kind == 1:
        skew = np.zeros((q + 1, q + 1), dtype=int)
        skew[0, 1:] = 1
        skew[1:, 0] = -1
        skew[1:, 1:] = jacobsthal
        return np.eye(q + 1, dtype=int) + skew
    conference = np.zeros((q + 1, q + 1), dtype=int)
    conference[0, 1:] = 1
    conference[1:, 0] = 1
    conference[1:, 1:] = jacobsthal
    return np.kron(conference, np.array([[1, 1], [1, -1]])) + np.kron(
        np.eye(q + 1, dtype=int), np.array([[1, -1], [-1, -1]])
    )


def _base_hadamard(order: int) -> Optional[np.ndarray]:
    """A Paley Hadamard matrix of exactly ``order``, or None."""
    if order in (1, 2):
        return hadamard(order)
    if _is_prime(order - 1) and (order - 1) % 4 == 3:
        return _paley(order - 1, 1)
    if order % 2 == 0 and _is_prime(order // 2 - 1) and (order // 2 - 1) % 4 == 1:
        return _paley(order // 2 - 1, 2)
    return None


def hadamard_matrix(order: int) -> np.ndarray:
    """Hadamard matrix of ``order`` from Sylvester doubling of a Paley matrix.

    Column 0 is all ones (rows are sign-flipped to make it so).

    Raises:
        ValueError: If ``order`` is not ``2^k`` times ``1``, ``2``, ``q + 1``
            (``q`` prime, ``q = 3 mod 4``) or ``2 (q + 1)`` (``q`` prime,
            ``q = 1 mod 4``).
    """
    base, doublings = order, 0
    while True:
        matrix = _base_hadamard(base)
        if matrix is not None:
            break
        if base % 2:
            raise ValueError(f"No Sylvester or Paley Hadamard matrix of order {order}.")
        base //= 2
        doublings += 1
    for _ in range(doublings):
        matrix = np.kron(np.array([[1, 1], [1, -1]]), matrix)
    return matrix * matrix[:, :1]


def smallest_hadamard_order(n: int) -> int:
    """Smallest order >= ``n`` that :func:`hadamard_matrix` can build."""
    order = max(int(n), 1)
    while True:
        try:
            base = order
            while _base_hadamard(base) is None:
                if base % 2:
                    raise ValueError
                base //= 2
            return order
        except ValueError:
            order += 1


def _balanced_columns(pool: np.ndarray, blocked: np.ndarray, count: int) -> np.ndarray:
    """``count`` columns of ``pool``, least content in ``blocked`` first.

    Columns are taken in groups of equal content in the (orthonormal)
    ``blocked`` subspace, lowest first; within a group, column-pivoted QR
    of the residuals against everything chosen so far orders them, most
    independent first, and dependent ones are skipped.
    """
    content = np.round(np.linalg.norm(blocked.T @ pool, axis=0) / np.sqrt(pool.shape[0]), 9)
    basis = blocked
    chosen = []
    for level in np.unique(content):
        group = np.flatnonzero(content == level)
        residual = pool[:, group]
        for _ in range(2):  # Gram-Schmidt twice against the chosen span
            residual = residual - basis @ (basis.T @ residual)
        q, r, pivots = qr(residual, mode="economic", pivoting=True)
        independent = np.abs(np.diag(r)) > 1e-6 * np.sqrt(pool.shape[0])
        take = min(int(np.count_nonzero(independent)), count - len(chosen))
        chosen.extend(group[pivots[:take]])
        basis = np.hstack([basis, q[:, :take]])
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise ValueError(f"Only {len(chosen)} independent Hadamard columns are available; requested {count}.")
    return pool[:, chosen]


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
        construction: str = "sylvester",
        selection: str = "first",
        check_rank: bool = True,
    ) -> np.ndarray:
        """
        Generate Hadamard modes (float entries of +1/-1).

        A Hadamard matrix of order at least the number of actuators is
        truncated to the actuators (rows) and ``n_modes`` columns. The
        truncated columns are generally neither orthogonal nor zero-mean;
        ``selection="balanced"`` picks columns that are close to both, and
        ``orthonormalize=True`` makes them exactly orthonormal (the entries
        are then no longer +/-1).

        Args:
            n_modes: Number of modes, at most the number of actuators minus
                the number of removed modes.
            ignore_piston: Remove piston: column 0 (all ones) is skipped and
                the mean is subtracted from the other columns, which are not
                zero-mean once truncated. Entries are then no longer +/-1
                (nearly so with ``selection="balanced"``).
            orthonormalize: Gram-Schmidt the modes in order so they are
                orthonormal on the actuator grid.
            remove: Further modes to project out (see
                :func:`aobasis.removal_basis`).
            normalize: Scale each mode to unit ``"rms"``, ``"l2"``,
                ``"peak"`` or ``"pv"`` after everything else (see
                :func:`aobasis.normalize_modes`). ``None`` keeps the
                generator's own scale.
            construction: ``"sylvester"``: the Sylvester matrix of the next
                power of two. ``"smallest"``: the smallest order that a
                Sylvester doubling of a Paley matrix reaches (e.g. 104
                instead of 128 for 97 actuators), so fewer rows are cut.
            selection: ``"first"``: columns in matrix order. ``"balanced"``:
                piston (unless removed) first, then the other columns in
                order of increasing piston (and removed-mode) content, the
                most mutually independent first among equally balanced ones
                (column-pivoted QR), skipping dependent ones. This trades
                some conditioning for balance (the Sylvester order is already
                about as well-conditioned as truncation allows);
                ``construction="smallest"`` usually improves both.
            check_rank: Warn if the modes are linearly dependent on the
                actuators. ``False`` skips the check, which is a large part
                of the cost for big bases (a column-pivoted QR of the modes);
                the modes are identical either way.
        """
        self._record_options(
            n_modes=n_modes,
            ignore_piston=ignore_piston,
            orthonormalize=orthonormalize,
            remove=remove,
            normalize=normalize,
            construction=construction,
            selection=selection,
        )
        _check_normalize(normalize)
        if construction not in CONSTRUCTIONS:
            raise ValueError(f"construction must be one of {CONSTRUCTIONS}, got {construction!r}.")
        if selection not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}, got {selection!r}.")
        removed = self._removed_subspace(remove, ignore_piston)
        n_modes = self._validate_n_modes(n_modes, max_modes=self.n_actuators - removed.shape[1])
        if n_modes == 0:
            self.modes = np.zeros((self.n_actuators, 0), dtype=float)
            return self.modes

        if construction == "sylvester":
            size = 1
            while size < self.n_actuators:
                size *= 2
            H = hadamard(size)
        else:
            size = smallest_hadamard_order(self.n_actuators)
            H = hadamard_matrix(size)
        H = H[: self.n_actuators].astype(float)

        if selection == "first":
            modes = self._take_outside(
                lambda start, count: H[:, start : start + count], n_modes, removed, n_available=size
            )
        else:
            piston = np.ones(self.n_actuators)
            keep_piston = np.linalg.norm(piston - removed @ (removed.T @ piston)) > 1e-8 * np.sqrt(
                self.n_actuators
            )
            chosen = _balanced_columns(
                H[:, 1:],
                removal_basis(self.positions, removed, ignore_piston=True),
                n_modes - int(keep_piston),
            )
            modes = np.hstack([H[:, :1], chosen]) if keep_piston else chosen
        return self._finish(
            modes, orthonormalize=orthonormalize, removed=removed, normalize=normalize, check_rank=check_rank
        )
