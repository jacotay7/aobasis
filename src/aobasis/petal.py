"""Petal (fragment-piston) modes of a pupil split into disconnected fragments.

Thick spiders (the ELT's six petals), segment gaps (the GMT's seven segments)
or any other mask that cuts the pupil into islands leave the wavefront free
to take a different piston on each island. Those pistons are invisible to a
wavefront sensor that only measures gradients inside the islands; they are
the "petal" or "island" modes of the low-wind effect and of segment phasing.
"""

import warnings
from typing import Optional

import numpy as np
from scipy import ndimage

from .utils import _validate_non_negative_integer

# 4-connectivity: pixels that only touch at a corner are not connected.
_FOUR_CONNECTED = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def label_fragments(pupil: np.ndarray, threshold: float = 1.0, min_pixels: int = 1):
    """Label the connected fragments of a pupil.

    Pixels with ``pupil >= threshold`` are lit; two lit pixels belong to the
    same fragment when a path of lit pixels joins them through shared edges
    (4-connectivity, so a diagonal contact does not join two fragments).
    Fragments with fewer than ``min_pixels`` pixels are dropped (their
    pixels get label 0).

    Fragments are numbered ``1 .. n`` in row-major (raster) order of their
    first pixel, the order of :func:`scipy.ndimage.label`, which depends only
    on the mask and is therefore the same on every machine.

    Args:
        pupil: 2-D pupil amplitude or mask, values in ``[0, 1]`` (booleans
            work too).
        threshold: Lowest value of a lit pixel. The default ``1.0`` keeps
            only fully lit pixels, so a spider or gap that grazes a pixel
            separates the fragments on either side of it, and the modes are
            uniform-weighted over pixels of equal transmission. For a pupil
            resampled with round-off, use e.g. ``0.999``; ``0.5`` also keeps
            half-lit edge pixels, which can bridge a spider thinner than two
            pixels.
        min_pixels: Smallest fragment kept, in pixels (``1`` keeps every
            fragment). Raise it to drop slivers left where spiders meet the
            obscuration or the rim.

    Returns:
        ``(labels, n_fragments)``: an integer ``[y, x]`` array of the same
        shape as ``pupil`` (0 outside every kept fragment), and the number
        of fragments.
    """
    pupil = np.asarray(pupil)
    if pupil.ndim != 2:
        raise ValueError(f"pupil must be 2-D, got shape {pupil.shape}.")
    pupil = pupil.astype(float)
    if not np.all(np.isfinite(pupil)):
        raise ValueError("pupil must contain only finite values.")
    if not np.isscalar(threshold) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("threshold must be a positive finite scalar.")
    min_pixels = _validate_non_negative_integer(min_pixels, "min_pixels", minimum=1)

    labels, n_found = ndimage.label(pupil >= threshold, structure=_FOUR_CONNECTED)
    if n_found == 0:
        return labels, 0
    sizes = np.bincount(labels.ravel(), minlength=n_found + 1)
    keep = sizes >= min_pixels
    keep[0] = False
    relabel = np.zeros(n_found + 1, dtype=labels.dtype)
    relabel[keep] = np.arange(1, int(keep.sum()) + 1)
    return relabel[labels], int(keep.sum())


class PetalBasis:
    """Piston modes of the disconnected fragments of a pupil.

    The pupil is split into fragments by :func:`label_fragments`. With ``n``
    fragments there are ``n`` raw piston indicators and ``n - 1`` petal
    modes: the indicators with global piston removed, orthonormalized over
    the lit pupil.

    The petal modes are unit RMS over the lit pixels, zero mean, and
    mutually orthogonal (``(modes * modes[k]).sum((1, 2)) / n_lit`` is the
    ``k``-th unit vector). They are the Gram-Schmidt orthonormalization of
    ``[piston, fragment 1, ..., fragment n - 1]`` with the piston dropped,
    so mode ``k`` (counting from 1) is positive on fragment ``k``, a negative
    constant on fragments ``k + 1 .. n`` and zero on fragments ``1 .. k - 1``
    (a Helmert basis weighted by fragment area). The ordering and signs are
    fixed, so the same pupil gives the same modes everywhere.

    Coordinates follow the stack convention: images are indexed ``[y, x]``.

    Args:
        pupil: 2-D pupil amplitude or mask, values in ``[0, 1]``.
        threshold: Lowest value of a lit pixel (see :func:`label_fragments`;
            default ``1.0``, fully lit pixels only).
        min_pixels: Smallest fragment kept, in pixels (default ``1``).

    Attributes:
        labels: Integer ``[y, x]`` fragment map, 0 outside every fragment.
        n_fragments: Number of fragments ``n``.
        lit: Boolean map of the pixels in a fragment.
        areas: Pixels per fragment, shape ``(n,)``.

    Example::

        petals = PetalBasis(pupil, min_pixels=20)
        petals.modes.shape                          # (n - 1, ny, nx)
        smooth = zernike_modes_on_mask(petals.lit, 6, pupil_radius=nx / 2)
        pistons = petals.petal_pistons(phase, smooth_modes=smooth)
    """

    def __init__(self, pupil: np.ndarray, threshold: float = 1.0, min_pixels: int = 1):
        self.labels, self.n_fragments = label_fragments(pupil, threshold=threshold, min_pixels=min_pixels)
        if self.n_fragments == 0:
            raise ValueError(
                "No lit fragment: no pixel reaches the threshold "
                f"({threshold}) in a fragment of at least min_pixels ({min_pixels}) pixels."
            )
        self.threshold = float(threshold)
        self.min_pixels = int(min_pixels)
        self.lit = self.labels > 0
        self.areas = np.bincount(self.labels.ravel(), minlength=self.n_fragments + 1)[1:]
        self._lit_labels = self.labels[self.lit] - 1  # fragment index of each lit pixel
        self._mode_coefficients = self._helmert_coefficients()

    @property
    def shape(self):
        """Shape ``(ny, nx)`` of the pupil."""
        return self.labels.shape

    @property
    def n_modes(self) -> int:
        """Number of petal modes, ``n_fragments - 1``."""
        return self.n_fragments - 1

    def _helmert_coefficients(self) -> np.ndarray:
        """``(n, n - 1)`` per-fragment values of the orthonormal petal modes.

        Mode ``j`` (from 0) is ``c W_j`` on fragment ``j`` and ``-c w_j`` on
        every later fragment, with ``w`` the fragment areas as fractions of
        the lit pupil, ``W_j`` the fraction in the fragments after ``j`` and
        ``c = 1 / sqrt(w_j W_j (w_j + W_j))``: zero mean, unit mean square,
        and orthogonal to the other modes. It is what Gram-Schmidt gives
        from ``[piston, fragment 1, ..., fragment n - 1]``, with exact zeros.
        """
        n = self.n_fragments
        weights = self.areas / self.areas.sum()
        after = np.concatenate([np.cumsum(weights[::-1])[::-1][1:], [0.0]])
        coefficients = np.zeros((n, n - 1))
        for j in range(n - 1):
            scale = 1.0 / np.sqrt(weights[j] * after[j] * (weights[j] + after[j]))
            coefficients[j, j] = scale * after[j]
            coefficients[j + 1 :, j] = -scale * weights[j]
        return coefficients

    def _images(self, per_fragment: np.ndarray) -> np.ndarray:
        """Scatter ``(n, k)`` per-fragment values to ``(k, ny, nx)`` images."""
        images = np.zeros((per_fragment.shape[1],) + self.shape)
        images[:, self.lit] = per_fragment[self._lit_labels].T
        return images

    @property
    def indicators(self) -> np.ndarray:
        """Raw piston of each fragment: ``(n, ny, nx)``, 1 on fragment ``k``, 0 elsewhere."""
        return self._images(np.eye(self.n_fragments))

    @property
    def modes(self) -> np.ndarray:
        """Orthonormal, piston-free petal modes: ``(n - 1, ny, nx)``, zero off the lit pupil."""
        return self._images(self._mode_coefficients)

    def piston_map(self, pistons: np.ndarray) -> np.ndarray:
        """The ``[y, x]`` map that has piston ``pistons[k]`` on fragment ``k``."""
        pistons = np.asarray(pistons, dtype=float)
        if pistons.shape != (self.n_fragments,):
            raise ValueError(f"pistons must have shape ({self.n_fragments},), got {pistons.shape}.")
        return self._images(pistons[:, None])[0]

    def petal_reconstructor(self, smooth_modes: Optional[np.ndarray] = None) -> np.ndarray:
        """Linear map from a phase map to its fragment pistons.

        Returns ``(n, ny, nx)`` weight images ``W`` (zero off the lit pupil)
        such that ``np.tensordot(W, phase, axes=2)`` equals
        :meth:`petal_pistons` ``(phase, smooth_modes)``. Build it once and
        reuse it for a stream of phase maps.

        Args:
            smooth_modes: Optional ``(m, ny, nx)`` stack of smooth modes
                (e.g. the first Zernikes from
                :func:`aobasis.zernike_modes_on_mask`) fitted jointly with the
                fragment pistons; see :meth:`petal_pistons`.
        """
        n_lit = int(self.lit.sum())
        design = np.zeros((n_lit, self.n_fragments))
        design[np.arange(n_lit), self._lit_labels] = 1.0
        if smooth_modes is not None:
            smooth = self._smooth_columns(smooth_modes)
            design = np.concatenate([design, smooth], axis=1)
            rank = np.linalg.matrix_rank(design)
            if rank < design.shape[1]:
                warnings.warn(
                    f"The smooth modes and fragment pistons are linearly dependent "
                    f"(rank {rank} < {design.shape[1]}): some combination of smooth "
                    "modes is a pattern of fragment pistons, so the pistons are not "
                    "unique. Use fewer or smoother modes.",
                    RuntimeWarning,
                    stacklevel=3,
                )
        fit = np.linalg.pinv(design)[: self.n_fragments]  # (n, n_lit) fragment-piston rows
        # Subtract the area-weighted mean piston (the global piston).
        weights = self.areas / self.areas.sum()
        fit = fit - weights @ fit
        weight_images = np.zeros((self.n_fragments,) + self.shape)
        weight_images[:, self.lit] = fit
        return weight_images

    def _smooth_columns(self, smooth_modes: np.ndarray) -> np.ndarray:
        smooth_modes = np.asarray(smooth_modes, dtype=float)
        if smooth_modes.ndim == 2:
            smooth_modes = smooth_modes[None]
        if smooth_modes.ndim != 3 or smooth_modes.shape[1:] != self.shape:
            raise ValueError(f"smooth_modes must have shape (m, {self.shape[0]}, {self.shape[1]}), got {smooth_modes.shape}.")
        columns = smooth_modes[:, self.lit].T  # (n_lit, m)
        # The fragment pistons already span the global piston: take it out of
        # the smooth modes and drop any that were piston only.
        norms = np.linalg.norm(columns, axis=0)
        columns = columns - columns.mean(axis=0)
        keep = np.linalg.norm(columns, axis=0) > 1e-10 * np.maximum(norms, np.finfo(float).tiny)
        return columns[:, keep]

    def petal_pistons(self, phase: np.ndarray, smooth_modes: Optional[np.ndarray] = None) -> np.ndarray:
        """Piston of each fragment, global piston removed.

        Without ``smooth_modes`` the piston of fragment ``k`` is the mean of
        ``phase`` over it. That counts smooth aberrations as petal: a tilt
        across the pupil has a different mean on every fragment, so the
        fragment means of a pure tip/tilt are not zero, and a petal sensor or
        controller fed with them would "correct" tilt by phasing the petals.
        With ``smooth_modes`` the fragment pistons and the smooth modes are
        fitted jointly by least squares over the lit pixels, so tip, tilt
        and whatever else the smooth modes span are absorbed by the smooth
        modes and only the step pattern is reported. Give only a few smooth,
        low-order modes: a high-order basis can itself approximate steps
        between fragments (a warning is raised if it can reproduce one
        exactly). A piston among the smooth modes is ignored.

        The result has its area-weighted mean removed, so
        ``piston_map(p)`` is zero-mean over the lit pupil, and the
        coefficients on the petal modes are
        ``(modes * piston_map(p)).sum((1, 2)) / lit.sum()``.

        Args:
            phase: ``(ny, nx)`` phase or OPD map, or a ``(k, ny, nx)`` stack.
                Only the lit pixels are used.
            smooth_modes: Optional ``(m, ny, nx)`` smooth modes to fit jointly.

        Returns:
            ``(n,)`` pistons, or ``(k, n)`` for a stack, in the units of
            ``phase``.
        """
        phase = np.asarray(phase, dtype=float)
        if phase.shape[-2:] != self.shape or phase.ndim not in (2, 3):
            raise ValueError(f"phase must have shape ({self.shape[0]}, {self.shape[1]}) or (k, ...), got {phase.shape}.")
        weights = self.petal_reconstructor(smooth_modes)
        return phase[..., self.lit] @ weights[:, self.lit].T


def petal_modes(pupil: np.ndarray, threshold: float = 1.0, min_pixels: int = 1, raw: bool = False) -> np.ndarray:
    """Petal (fragment-piston) modes of a pupil, as a stack of images.

    A shortcut for :class:`PetalBasis` ``(pupil, threshold, min_pixels)``.

    Args:
        pupil: 2-D pupil amplitude or mask, values in ``[0, 1]``.
        threshold: Lowest value of a lit pixel (default ``1.0``).
        min_pixels: Smallest fragment kept, in pixels (default ``1``).
        raw: Return the ``n`` raw fragment indicators instead of the
            ``n - 1`` orthonormal piston-free modes.

    Returns:
        ``(n - 1, ny, nx)`` modes (``(n, ny, nx)`` with ``raw=True``), zero
        off the lit pupil. A pupil in one piece gives ``(0, ny, nx)``.
    """
    basis = PetalBasis(pupil, threshold=threshold, min_pixels=min_pixels)
    return basis.indicators if raw else basis.modes
