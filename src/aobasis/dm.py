"""A deformable-mirror model: Gaussian influence functions evaluated on a pupil.

:class:`DeformableMirror` turns actuator commands into a surface sampled on a
regular pixel grid or at arbitrary points, with the influence functions of
:func:`aobasis.gaussian_influence_functions`. On a pixel grid with actuators
on a rectangular lattice (any square grid, including one clipped by a pupil)
it uses the separability of the Gaussian,

    coupling ** (((x - x_a)^2 + (y - y_a)^2) / pitch^2)
        = coupling ** ((x - x_a)^2 / pitch^2) * coupling ** ((y - y_a)^2 / pitch^2),

so that ``surface = G_y @ C @ G_x.T``, with ``C`` the commands scattered
onto the lattice. That equals ``influence_matrix() @ cmd`` to round-off,
needs no ``(n_pixels, n_actuators)`` matrix, and is much faster (for 877
actuators on 144 x 144 pixels, 0.19 ms instead of 4.3 ms, or 14 ms on one
thread; ``examples/dm_surface.py``).

:meth:`DeformableMirror.transformed` returns the same DM shifted, rotated and
magnified in the pupil, for misregistration studies (a "true" DM against the
calibrated one).
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from .base import _validate_positions_array
from .influence import _nearest_neighbour_distance, gaussian_influence_functions
from .utils import _validate_positive_finite_scalar

_METHODS = ("auto", "separable", "matrix")


def _lattice_axis(values: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Distinct values of ``values`` (within ``tol``) and each value's index into them."""
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    first = np.ones(len(values), dtype=bool)
    first[1:] = np.diff(ordered) > tol
    index = np.empty(len(values), dtype=np.intp)
    index[order] = np.cumsum(first) - 1
    return ordered[first], index


def _axis(values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"grid {name} axis must be a non-empty 1-D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"grid {name} axis must be finite.")
    return values


class DeformableMirror:
    """A DM with Gaussian influence functions, sampled on a grid or at points.

    Actuator ``a`` with command ``c_a`` contributes
    ``c_a * coupling ** (|p - x_a|^2 / pitch^2)`` at point ``p``: 1 at the
    actuator and ``coupling`` one pitch away, exactly as
    :func:`aobasis.gaussian_influence_functions`. The surface is in the units
    of the commands; no reflection factor is applied (a mirror displacement
    ``s`` gives OPD ``2 s`` at normal incidence).

    Args:
        positions: ``(n_actuators, 2)`` actuator coordinates ``(x, y)`` in
            metres, in the pupil plane. Commands follow this order.
        pitch: Actuator pitch in metres; by default the median
            nearest-neighbour distance between actuators.
        coupling: Inter-actuator coupling, ``0 < coupling < 1``.
        grid: ``(y, x)``: 1-D pixel-centre coordinates (metres) of the rows
            and columns of a pixel grid. :meth:`surface` then returns
            ``(ny, nx)`` images indexed ``[y, x]``. For ``n`` pixels of
            size ``d`` centred on the optical axis (the AO stack
            convention) use ``(np.arange(n) - (n - 1) / 2) * d`` for both.
        points: ``(n_points, 2)`` arbitrary sample points ``(x, y)``
            instead of a grid; :meth:`surface` then returns ``(n_points,)``
            vectors.

    Exactly one of ``grid`` and ``points`` is required.

    Example:
        >>> import numpy as np, aobasis
        >>> actuators = aobasis.make_circular_actuator_grid(8.0, 21)
        >>> axis = (np.arange(64) - 31.5) * (8.0 / 64)
        >>> dm = aobasis.DeformableMirror(actuators, pitch=0.4, grid=(axis, axis))
        >>> dm.surface(np.ones(dm.n_actuators)).shape
        (64, 64)
        >>> shifted = dm.transformed(shift=(0.05, 0.0), rotation_deg=1.0)
    """

    def __init__(
        self,
        positions: np.ndarray,
        pitch: Optional[float] = None,
        coupling: float = 0.15,
        *,
        grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        points: Optional[np.ndarray] = None,
    ):
        positions = _validate_positions_array(positions)
        if not np.isscalar(coupling) or not np.isfinite(coupling) or not 0 < coupling < 1:
            raise ValueError("coupling must be in (0, 1).")
        self.positions = positions.copy()
        self.positions.flags.writeable = False
        self.coupling = float(coupling)
        self.pitch = (
            _nearest_neighbour_distance(positions) if pitch is None else _validate_positive_finite_scalar(pitch, "pitch")
        )
        if (grid is None) == (points is None):
            raise ValueError("pass exactly one of grid=(y, x) and points.")
        if grid is not None:
            if len(grid) != 2:
                raise ValueError("grid must be a (y, x) pair of 1-D coordinate arrays.")
            self.grid: Optional[Tuple[np.ndarray, np.ndarray]] = (_axis(grid[0], "y"), _axis(grid[1], "x"))
            for axis in self.grid:
                axis.flags.writeable = False
            self._points: Optional[np.ndarray] = None
        else:
            self.grid = None
            self._points = _validate_positions_array(points).copy()
            self._points.flags.writeable = False
        self._influence: Optional[np.ndarray] = None
        self._factors: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
        self._lattice: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None
        self._separable = self._plan_separable()

    # Geometry -------------------------------------------------------------

    @property
    def n_actuators(self) -> int:
        """Number of actuators (length of a command vector)."""
        return self.positions.shape[0]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of one surface: ``(ny, nx)`` on a grid, ``(n_points,)`` at points."""
        if self.grid is not None:
            return (self.grid[0].size, self.grid[1].size)
        return (self._points.shape[0],)

    @property
    def points(self) -> np.ndarray:
        """``(n_points, 2)`` sample points ``(x, y)``; row-major pixel centres on a grid."""
        if self._points is not None:
            return self._points
        xx, yy = np.meshgrid(self.grid[1], self.grid[0])
        return np.column_stack((xx.ravel(), yy.ravel()))

    @property
    def separable(self) -> bool:
        """True when :meth:`surface` uses the separable fast path by default.

        That needs a pixel grid, actuators on a rectangular lattice (each
        lattice cell holding at most one actuator), and the separable product
        to cost fewer operations than ``influence_matrix() @ cmd``. A rotated
        DM (:meth:`transformed`) usually falls back to the matrix.
        """
        return self._separable

    def _plan_separable(self) -> bool:
        """Find the actuator lattice; decide whether the separable path pays off."""
        if self.grid is None:
            return False
        scale = max(self.pitch, float(np.max(np.abs(self.positions))))
        tol = 1e-12 * scale
        ux, ix = _lattice_axis(self.positions[:, 0], tol)
        uy, iy = _lattice_axis(self.positions[:, 1], tol)
        cells = iy * ux.size + ix
        if np.unique(cells).size != cells.size:  # two actuators in one lattice cell
            return False
        self._lattice = (uy, ux, iy, ix)
        ny, nx = self.shape
        separable_cost = min(ny * uy.size * ux.size + ny * ux.size * nx, uy.size * ux.size * nx + ny * uy.size * nx)
        return separable_cost <= ny * nx * self.n_actuators

    def _separable_factors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._factors is None:
            if self._lattice is None:
                raise ValueError("the actuators are not on a lattice with one actuator per cell.")
            uy, ux, iy, ix = self._lattice
            log_c = np.log(self.coupling)
            gy = np.exp(log_c * (self.grid[0][:, None] - uy[None, :]) ** 2 / self.pitch**2)
            gx = np.exp(log_c * (self.grid[1][:, None] - ux[None, :]) ** 2 / self.pitch**2)
            self._factors = (gy, gx, iy, ix)
        return self._factors

    # Surfaces -------------------------------------------------------------

    def influence_matrix(self) -> np.ndarray:
        """``(n_points, n_actuators)`` influence functions at :attr:`points`.

        The same values as :func:`aobasis.gaussian_influence_functions`; on a
        grid the rows are the pixels in row-major (``[y, x]``) order. The
        matrix is built on first use and cached (read-only; copy it to modify).
        """
        if self._influence is None:
            influence = gaussian_influence_functions(self.positions, self.points, self.coupling, self.pitch)
            influence.flags.writeable = False
            self._influence = influence
        return self._influence

    def surface(self, cmd: np.ndarray, method: str = "auto") -> np.ndarray:
        """DM surface for one command vector or a batch of them.

        Args:
            cmd: ``(n_actuators,)`` commands, or ``(n_actuators, n_cmds)``
                with one command vector per column (like an M2C matrix).
            method: ``"auto"`` (separable when :attr:`separable`),
                ``"separable"`` (grid with actuators on a lattice only) or
                ``"matrix"`` (``influence_matrix() @ cmd``).

        Returns:
            On a grid, ``(ny, nx)`` for one command and ``(n_cmds, ny, nx)``
            for a batch. At points, ``(n_points,)`` and ``(n_points, n_cmds)``.
        """
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")
        cmd = np.asarray(cmd, dtype=float)
        if cmd.ndim not in (1, 2) or cmd.shape[0] != self.n_actuators:
            raise ValueError(f"cmd must have shape ({self.n_actuators},) or ({self.n_actuators}, n_cmds), got {cmd.shape}.")
        if method == "separable" and self.grid is None:
            raise ValueError("the separable path needs a pixel grid (grid=(y, x)).")
        if method == "separable" or (method == "auto" and self._separable):
            return self._surface_separable(cmd)
        out = self.influence_matrix() @ cmd
        if self.grid is None:
            return out
        if cmd.ndim == 1:
            return out.reshape(self.shape)
        return out.T.reshape((cmd.shape[1],) + self.shape)

    def _surface_separable(self, cmd: np.ndarray) -> np.ndarray:
        gy, gx, iy, ix = self._separable_factors()
        batch = cmd.reshape(self.n_actuators, -1)
        lattice = np.zeros((batch.shape[1], gy.shape[1], gx.shape[1]))
        lattice[:, iy, ix] = batch.T
        ny, nx = self.shape
        nuy, nux = lattice.shape[1:]
        if ny * nuy * nux + ny * nux * nx <= nuy * nux * nx + ny * nuy * nx:
            out = (gy @ lattice) @ gx.T
        else:
            out = gy @ (lattice @ gx.T)
        return out[0] if cmd.ndim == 1 else out

    # Misregistration ------------------------------------------------------

    def transformed(
        self,
        shift: Sequence[float] = (0.0, 0.0),
        rotation_deg: float = 0.0,
        magnification: float = 1.0,
    ) -> "DeformableMirror":
        """The same DM, misregistered in the pupil.

        Each actuator moves to ``magnification * R(rotation_deg) @ x_a + shift``,
        about the pupil origin, and the pitch (so the influence-function
        width) scales by ``magnification``. The sampling grid or points stay
        where they are. So the new surface at point ``p`` equals this DM's
        surface at ``R^-1 (p - shift) / magnification``.

        Args:
            shift: ``(dx, dy)`` in metres.
            rotation_deg: Rotation in degrees, counter-clockwise from +x
                towards +y.
            magnification: Scale factor (``> 0``) of the DM in the pupil.

        Returns:
            A new :class:`DeformableMirror` (commands keep their order).
        """
        shift = np.asarray(shift, dtype=float)
        if shift.shape != (2,) or not np.all(np.isfinite(shift)):
            raise ValueError("shift must be a finite (dx, dy) pair.")
        if not np.isscalar(rotation_deg) or not np.isfinite(rotation_deg):
            raise ValueError("rotation_deg must be a finite scalar.")
        magnification = _validate_positive_finite_scalar(magnification, "magnification")
        angle = np.deg2rad(rotation_deg)
        cos, sin = np.cos(angle), np.sin(angle)
        x, y = self.positions.T
        positions = np.column_stack(
            (magnification * (cos * x - sin * y) + shift[0], magnification * (sin * x + cos * y) + shift[1])
        )
        return DeformableMirror(
            positions,
            pitch=self.pitch * magnification,
            coupling=self.coupling,
            grid=self.grid,
            points=self._points,
        )

    def __repr__(self) -> str:
        where = f"grid {self.shape[0]}x{self.shape[1]}" if self.grid is not None else f"{self.shape[0]} points"
        return (
            f"DeformableMirror({self.n_actuators} actuators, pitch={self.pitch:g}, "
            f"coupling={self.coupling:g}, {where}, separable={self._separable})"
        )
