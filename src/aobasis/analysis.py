"""Helpers for using a basis: modal coefficients and a quality report."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import _numerical_rank


def _as_modes(modes: np.ndarray) -> np.ndarray:
    modes = np.asarray(modes, dtype=float)
    if modes.ndim != 2:
        raise ValueError("modes must have shape (n_actuators, n_modes).")
    if not np.all(np.isfinite(modes)):
        raise ValueError("modes must contain only finite values.")
    return modes


def command_to_mode_matrix(modes: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    """Pseudo-inverse of the modal-to-command matrix (C2M), ``(n_modes, n_actuators)``.

    ``C2M @ commands`` gives the least-squares modal coefficients of
    commands, and ``modes @ (C2M @ commands)`` their projection on the basis.
    Singular values below ``rcond`` times the largest are dropped.
    """
    return np.linalg.pinv(_as_modes(modes), rcond=rcond)


def fit_coefficients(
    modes: np.ndarray, commands: np.ndarray, rcond: float = 1e-10, regularization: float = 0.0
) -> np.ndarray:
    """Least-squares modal coefficients ``c`` with ``modes @ c ~ commands``.

    Args:
        modes: ``(n_actuators, n_modes)`` modal-to-command matrix.
        commands: ``(n_actuators,)`` or ``(n_actuators, n_samples)``.
        rcond: Relative cutoff for small singular values of ``modes``.
        regularization: Tikhonov weight, in units of the mean squared
            singular value of ``modes``.

    Returns:
        ``(n_modes,)`` or ``(n_modes, n_samples)`` coefficients.
    """
    modes = _as_modes(modes)
    commands = np.asarray(commands, dtype=float)
    vector = commands.ndim == 1
    if vector:
        commands = commands[:, None]
    if commands.ndim != 2 or commands.shape[0] != modes.shape[0]:
        raise ValueError(f"commands must have {modes.shape[0]} rows to match modes.")
    if not np.isscalar(regularization) or not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be a non-negative finite scalar.")
    u, s, vt = np.linalg.svd(modes, full_matrices=False)
    keep = s > rcond * s[0] if s.size and s[0] > 0 else np.zeros(s.shape, dtype=bool)
    lam = regularization * np.mean(s**2) if s.size else 0.0
    gain = np.zeros_like(s)
    gain[keep] = s[keep] / (s[keep] ** 2 + lam)
    coefficients = vt.T @ (gain[:, None] * (u.T @ commands))
    return coefficients[:, 0] if vector else coefficients


@dataclass
class BasisReport:
    """Summary of a modal basis (see :func:`basis_report`)."""

    n_actuators: int
    n_modes: int
    rank: int
    condition_number: float
    max_cosine: float
    piston_content: np.ndarray
    norm_range: tuple

    def __str__(self) -> str:
        lines = [
            f"{self.n_modes} modes on {self.n_actuators} actuators, rank {self.rank}",
            f"condition number {self.condition_number:.3g}",
            f"largest |cos| between two modes {self.max_cosine:.3g}",
            f"piston content (|mean| / RMS): max {self.piston_content.max():.3g}"
            if self.n_modes
            else "piston content: -",
            f"mode L2 norms {self.norm_range[0]:.3g} to {self.norm_range[1]:.3g}",
        ]
        return "\n".join(lines)


def basis_report(modes: np.ndarray) -> BasisReport:
    """Rank, conditioning, orthogonality and piston content of ``modes``.

    ``condition_number`` is ``inf`` when the numerical rank is below the
    number of modes. ``max_cosine`` is the largest ``|cos|`` of the angle between two
    different modes (0 for an orthogonal basis); ``piston_content`` is each
    mode's ``|mean| / RMS`` over the actuators (0 for zero-mean modes, 1 for
    piston).
    """
    modes = _as_modes(modes)
    n_act, n_modes = modes.shape
    norms = np.linalg.norm(modes, axis=0)
    if n_modes == 0:
        return BasisReport(n_act, 0, 0, 1.0, 0.0, np.zeros(0), (0.0, 0.0))
    rank = _numerical_rank(modes)
    s = np.linalg.svd(modes, compute_uv=False)
    # Numerically singular bases report an infinite condition number.
    condition = float(s[0] / s[-1]) if rank == min(modes.shape) == n_modes and s[-1] > 0 else float("inf")
    unit = modes / np.where(norms > 0, norms, 1.0)
    cosines = np.abs(unit.T @ unit)
    np.fill_diagonal(cosines, 0.0)
    rms = np.sqrt(np.mean(modes**2, axis=0))
    piston = np.abs(modes.mean(axis=0)) / np.where(rms > 0, rms, 1.0)
    return BasisReport(
        n_actuators=n_act,
        n_modes=n_modes,
        rank=rank,
        condition_number=condition,
        max_cosine=float(cosines.max()) if n_modes > 1 else 0.0,
        piston_content=piston,
        norm_range=(float(norms.min()), float(norms.max())),
    )
