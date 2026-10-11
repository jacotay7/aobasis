"""DeformableMirror: separable fast path, matrix path and misregistration."""

import numpy as np
import pytest

from aobasis import DeformableMirror, gaussian_influence_functions, make_circular_actuator_grid


def _pixel_axis(n, pixel):
    return (np.arange(n) - 0.5 * (n - 1)) * pixel


@pytest.fixture
def actuators():
    return make_circular_actuator_grid(8.0, 17)  # pitch 0.5 m, clipped by the pupil


@pytest.fixture
def dm(actuators):
    axis = _pixel_axis(48, 8.0 / 48)
    return DeformableMirror(actuators, pitch=0.5, coupling=0.2, grid=(axis, axis))


def _rel(a, b):
    return np.max(np.abs(a - b)) / np.max(np.abs(b))


@pytest.mark.parametrize("shape", [(48, 48), (37, 52)])
def test_separable_equals_influence_matrix(actuators, shape):
    grid = (_pixel_axis(shape[0], 0.19), _pixel_axis(shape[1], 0.17))
    dm = DeformableMirror(actuators, pitch=0.5, grid=grid)
    assert dm.separable
    cmd = np.random.default_rng(0).standard_normal(dm.n_actuators)
    fast = dm.surface(cmd)
    assert fast.shape == shape
    assert _rel(fast, dm.surface(cmd, method="matrix")) < 1e-13
    yy, xx = np.meshgrid(*grid, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    direct = gaussian_influence_functions(actuators, points, coupling=0.15, pitch=0.5) @ cmd
    assert _rel(fast.ravel(), direct) < 1e-13


def test_matrix_path_is_gaussian_influence_functions(dm, actuators):
    expected = gaussian_influence_functions(actuators, dm.points, coupling=0.2, pitch=0.5)
    assert np.array_equal(dm.influence_matrix(), expected)
    assert dm.influence_matrix() is dm.influence_matrix()  # cached
    assert not dm.influence_matrix().flags.writeable


def test_batch_of_commands(dm):
    cmds = np.random.default_rng(1).standard_normal((dm.n_actuators, 5))
    batch = dm.surface(cmds)
    assert batch.shape == (5,) + dm.shape
    for k in range(5):
        assert _rel(batch[k], dm.surface(cmds[:, k])) < 1e-14
    assert _rel(batch, dm.surface(cmds, method="matrix")) < 1e-13


def test_points_mode(actuators):
    points = np.random.default_rng(2).uniform(-4, 4, (300, 2))
    dm = DeformableMirror(actuators, pitch=0.5, points=points)
    assert not dm.separable and dm.shape == (300,)
    cmd = np.random.default_rng(3).standard_normal((dm.n_actuators, 3))
    influence = gaussian_influence_functions(actuators, points, pitch=0.5)
    assert np.array_equal(dm.surface(cmd), influence @ cmd)
    assert dm.surface(cmd[:, 0]).shape == (300,)
    with pytest.raises(ValueError, match="pixel grid"):
        dm.surface(cmd, method="separable")


def test_unit_command_peaks_at_the_actuator(actuators):
    axis = _pixel_axis(33, 0.25)  # pixel centres fall on the actuators
    dm = DeformableMirror(actuators, pitch=0.5, coupling=0.15, grid=(axis, axis))
    centre = int(np.argmin(np.hypot(*actuators.T)))
    cmd = np.zeros(dm.n_actuators)
    cmd[centre] = 1.0
    surface = dm.surface(cmd)
    assert np.unravel_index(np.argmax(surface), surface.shape) == (16, 16)
    assert surface[16, 16] == pytest.approx(1.0)
    assert surface[16, 18] == pytest.approx(0.15)  # one pitch along +x
    assert surface[18, 16] == pytest.approx(0.15)


def test_default_pitch_is_nearest_neighbour_distance(actuators):
    dm = DeformableMirror(actuators, points=actuators)
    assert dm.pitch == pytest.approx(0.5)


def test_integer_pixel_shift_moves_the_image(dm):
    pixel = 8.0 / 48
    cmd = np.random.default_rng(4).standard_normal(dm.n_actuators)
    moved = dm.transformed(shift=(2 * pixel, -pixel))
    assert moved.separable  # a shift keeps the lattice
    s, t = dm.surface(cmd), moved.surface(cmd)
    # +2 pixels along x (columns), -1 along y (rows): t[i, j] = s[i + 1, j - 2]
    assert _rel(t[:-1, 2:], s[1:, :-2]) < 1e-12


def test_rotation_by_90_degrees_rotates_the_image(dm):
    cmd = np.random.default_rng(5).standard_normal(dm.n_actuators)
    rotated = dm.transformed(rotation_deg=90.0)
    # t(x, y) = s(y, -x): t[i, j] = s[n - 1 - j, i]
    assert _rel(rotated.surface(cmd), dm.surface(cmd)[::-1, :].T) < 1e-12


@pytest.mark.parametrize(
    "shift, rotation, magnification",
    [((0.13, -0.07), 0.0, 1.0), ((0.0, 0.0), 7.5, 1.0), ((0.0, 0.0), 0.0, 1.04), ((0.21, 0.05), -3.0, 0.97)],
)
def test_transform_matches_analytic_inverse(actuators, shift, rotation, magnification):
    points = np.random.default_rng(6).uniform(-4, 4, (500, 2))
    cmd = np.random.default_rng(7).standard_normal(len(actuators))
    true = DeformableMirror(actuators, pitch=0.5, points=points).transformed(shift, rotation, magnification)
    # surface of the transformed DM at p = surface of the original at T^-1 p
    angle = np.deg2rad(rotation)
    q = (points - np.asarray(shift)) / magnification
    back = np.column_stack(
        (np.cos(angle) * q[:, 0] + np.sin(angle) * q[:, 1], -np.sin(angle) * q[:, 0] + np.cos(angle) * q[:, 1])
    )
    expected = DeformableMirror(actuators, pitch=0.5, points=back).surface(cmd)
    assert _rel(true.surface(cmd), expected) < 1e-12
    assert true.pitch == pytest.approx(0.5 * magnification)


def test_rotated_dm_falls_back_to_the_matrix(dm):
    rotated = dm.transformed(rotation_deg=10.0)
    assert not rotated.separable
    cmd = np.random.default_rng(8).standard_normal(dm.n_actuators)
    # forcing the separable path on an off-lattice DM still works (it is
    # exact, only slower), and agrees
    assert _rel(rotated.surface(cmd, method="separable"), rotated.surface(cmd)) < 1e-12


def test_random_and_duplicate_actuators_are_not_separable():
    axis = _pixel_axis(20, 0.1)
    random = np.random.default_rng(9).uniform(-1, 1, (50, 2))
    assert not DeformableMirror(random, pitch=0.3, grid=(axis, axis)).separable
    duplicated = np.array([[0.0, 0.0], [0.0, 0.0], [0.5, 0.0]])
    dm = DeformableMirror(duplicated, pitch=0.5, grid=(axis, axis))
    assert not dm.separable
    with pytest.raises(ValueError, match="lattice"):
        dm.surface(np.ones(3), method="separable")
    assert dm.surface(np.ones(3)).shape == (20, 20)


def test_surface_is_linear(dm):
    rng = np.random.default_rng(10)
    a, b = rng.standard_normal((2, dm.n_actuators))
    assert _rel(dm.surface(2 * a - b), 2 * dm.surface(a) - dm.surface(b)) < 1e-12


def test_points_property_and_repr(dm):
    assert dm.points.shape == (48 * 48, 2)
    assert np.array_equal(dm.points[1], [dm.grid[1][1], dm.grid[0][0]])  # row-major, (x, y)
    assert "separable=True" in repr(dm)
    assert "points" in repr(DeformableMirror(dm.positions, pitch=0.5, points=dm.points[:10]))


def test_validation(actuators):
    axis = _pixel_axis(8, 1.0)
    with pytest.raises(ValueError, match="exactly one"):
        DeformableMirror(actuators, pitch=0.5)
    with pytest.raises(ValueError, match="exactly one"):
        DeformableMirror(actuators, pitch=0.5, grid=(axis, axis), points=actuators)
    with pytest.raises(ValueError, match="coupling"):
        DeformableMirror(actuators, pitch=0.5, coupling=1.0, grid=(axis, axis))
    with pytest.raises(ValueError, match="pitch"):
        DeformableMirror(actuators, pitch=-1.0, grid=(axis, axis))
    with pytest.raises(ValueError, match="grid"):
        DeformableMirror(actuators, pitch=0.5, grid=(axis,))
    with pytest.raises(ValueError, match="1-D"):
        DeformableMirror(actuators, pitch=0.5, grid=(axis, np.zeros((2, 2))))
    with pytest.raises(ValueError, match="finite"):
        DeformableMirror(actuators, pitch=0.5, grid=(axis, axis * np.nan))
    dm = DeformableMirror(actuators, pitch=0.5, grid=(axis, axis))
    with pytest.raises(ValueError, match="cmd"):
        dm.surface(np.ones(3))
    with pytest.raises(ValueError, match="method"):
        dm.surface(np.ones(dm.n_actuators), method="fft")
    with pytest.raises(ValueError, match="shift"):
        dm.transformed(shift=(1.0,))
    with pytest.raises(ValueError, match="rotation"):
        dm.transformed(rotation_deg=np.inf)
    with pytest.raises(ValueError, match="magnification"):
        dm.transformed(magnification=0.0)
