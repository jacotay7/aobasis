"""Mathematical properties of the generated bases (Noll order, rank, dtype, piston)."""

import warnings

import numpy as np
import pytest

from aobasis import (
    BasisGenerator,
    ConcreteBasis,
    FourierBasisGenerator,
    HadamardBasisGenerator,
    KLBasisGenerator,
    ZernikeBasisGenerator,
    ZonalBasisGenerator,
    ZonalFastBasisGenerator,
    make_circular_actuator_grid,
    orthonormalize_modes,
    positions_from_mask,
)


@pytest.fixture
def grid():
    return make_circular_actuator_grid(telescope_diameter=10.0, grid_size=12)


NOLL = {
    1: (0, 0), 2: (1, 1), 3: (1, -1), 4: (2, 0), 5: (2, -2), 6: (2, 2),
    7: (3, -1), 8: (3, 1), 9: (3, -3), 10: (3, 3), 11: (4, 0), 12: (4, 2),
    13: (4, -2), 14: (4, 4), 15: (4, -4), 16: (5, 1), 17: (5, -1), 18: (5, 3),
    19: (5, -3), 20: (5, 5), 21: (5, -5), 22: (6, 0),
}


def test_noll_to_nm_matches_noll_table():
    gen = ZernikeBasisGenerator(np.zeros((1, 2)), pupil_radius=1.0)
    assert {j: gen._noll_to_nm(j) for j in NOLL} == NOLL
    with pytest.raises(ValueError):
        gen._noll_to_nm(0)


def test_zernike_modes_are_noll_normalized():
    # A fine grid approximates the continuous disk, where Noll Zernikes have unit RMS.
    positions = make_circular_actuator_grid(2.0, 81)
    modes = ZernikeBasisGenerator(positions, pupil_radius=1.0).generate(10, ignore_piston=True)
    rms = np.sqrt(np.mean(modes**2, axis=0))
    assert np.allclose(rms, 1.0, atol=0.03)
    # j=5 is sin(2 theta): largest along the diagonal, zero on the axes.
    x, y = positions[:, 0], positions[:, 1]
    astig = modes[:, 3]
    assert np.allclose(astig[(np.abs(y) < 1e-9)], 0.0, atol=1e-9)


def test_zernike_radial_is_stable_at_high_order():
    # The explicit factorial sum gave R_48^0(1) = 3 and R_60^0(1) = 193587.
    gen = ZernikeBasisGenerator(np.zeros((1, 2)), pupil_radius=1.0)
    x, w = np.polynomial.legendre.leggauss(200)
    rho, w = (x + 1) / 2, w / 2
    for m in (0, 1, 7):
        orders = np.arange(m, 120, 2)
        radial = np.array([gen._zernike_radial(n, m, rho) for n in orders])
        assert np.allclose([gen._zernike_radial(n, m, np.array([1.0]))[0] for n in orders], 1.0)
        gram = (radial * w * rho) @ radial.T * 2 * (orders[:, None] + 1)
        assert np.allclose(gram, np.eye(len(orders)), atol=1e-10)


def test_zernike_radial_matches_explicit_formula_at_low_order():
    from math import factorial

    gen = ZernikeBasisGenerator(np.zeros((1, 2)), pupil_radius=1.0)
    rho = np.linspace(0.0, 1.0, 11)
    for n in range(12):
        for m in range(n % 2, n + 1, 2):
            expected = sum(
                (-1) ** k * factorial(n - k)
                / (factorial(k) * factorial((n + m) // 2 - k) * factorial((n - m) // 2 - k))
                * rho ** (n - 2 * k)
                for k in range((n - m) // 2 + 1)
            )
            assert np.allclose(gen._zernike_radial(n, m, rho), expected, atol=1e-12)


def test_large_zernike_basis_stays_bounded():
    positions = make_circular_actuator_grid(telescope_diameter=10.0, grid_size=50)
    with warnings.catch_warnings():  # high orders genuinely alias on the grid
        warnings.simplefilter("ignore", RuntimeWarning)
        modes = ZernikeBasisGenerator(positions, pupil_radius=5.0).generate(1500)  # j <= 1500, so n <= 54
    n_max = 54
    assert np.abs(modes).max() <= np.sqrt(2 * (n_max + 1)) + 1e-9


def test_zernike_warns_for_actuators_outside_pupil(grid):
    with pytest.warns(RuntimeWarning, match="outside pupil_radius"):
        modes = ZernikeBasisGenerator(grid, pupil_radius=2.0).generate(4)
    rho = np.linalg.norm(grid, axis=1) / 2.0
    # Defocus is extrapolated at the true radius, not clipped at rho = 1.
    assert np.allclose(modes[:, 3], np.sqrt(3) * (2 * rho**2 - 1))


def test_zernike_rejects_more_modes_than_actuators_and_warns_on_rank(grid):
    gen = ZernikeBasisGenerator(grid, pupil_radius=5.0)
    with pytest.raises(ValueError):
        gen.generate(grid.shape[0] + 1)
    with pytest.warns(RuntimeWarning, match="linearly dependent"):
        gen.generate(grid.shape[0])


def test_fourier_basis_has_full_rank_or_raises(grid):
    n = grid.shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        modes = FourierBasisGenerator(grid, pupil_diameter=10.0).generate(n)
    assert np.linalg.matrix_rank(modes) == n
    assert np.allclose(modes[:, 0], 1.0)

    # On an odd grid spanning the pupil, 1/D frequency steps cannot span every actuator.
    odd = make_circular_actuator_grid(10.0, 11)
    with pytest.raises(ValueError, match="independent Fourier modes"):
        FourierBasisGenerator(odd, pupil_diameter=10.0).generate(odd.shape[0])


def test_fourier_without_piston_is_independent_of_piston(grid):
    n = grid.shape[0] - 1
    modes = FourierBasisGenerator(grid, pupil_diameter=10.0).generate(n, ignore_piston=True)
    with_piston = np.column_stack((np.ones(grid.shape[0]), modes))
    assert np.linalg.matrix_rank(with_piston) == n + 1


def test_hadamard_is_float_and_supports_ignore_piston(grid):
    gen = HadamardBasisGenerator(grid)
    modes = gen.generate(8)
    assert modes.dtype == np.float64
    assert np.all(modes[:, 0] == 1.0)
    no_piston = gen.generate(8, ignore_piston=True)
    assert np.array_equal(no_piston, modes[:, 1:9]) if modes.shape[1] > 8 else True
    assert not np.all(no_piston[:, 0] == 1.0)
    with pytest.raises(ValueError):
        gen.generate(grid.shape[0] + 1)
    with pytest.raises(ValueError):
        gen.generate(grid.shape[0], ignore_piston=True)


@pytest.mark.parametrize(
    "factory",
    [
        lambda p: ZernikeBasisGenerator(p, pupil_radius=5.0),
        lambda p: FourierBasisGenerator(p, pupil_diameter=10.0),
        lambda p: HadamardBasisGenerator(p),
    ],
)
def test_orthonormalize_option(grid, factory):
    raw = factory(grid).generate(30)
    modes = factory(grid).generate(30, orthonormalize=True)
    assert np.allclose(modes.T @ modes, np.eye(30), atol=1e-10)
    # Gram-Schmidt keeps the order: mode k is in the span of raw modes 0..k.
    for k in (0, 5, 29):
        span = raw[:, : k + 1]
        coeffs, *_ = np.linalg.lstsq(span, modes[:, k], rcond=None)
        assert np.allclose(span @ coeffs, modes[:, k], atol=1e-10)
    # First mode keeps its sign.
    assert modes[:, 0] @ raw[:, 0] > 0


def test_kl_ignore_piston_modes_have_zero_mean(grid):
    gen = KLBasisGenerator(grid)
    modes = gen.generate(grid.shape[0] - 1, ignore_piston=True)
    assert np.allclose(modes.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(modes.T @ modes, np.eye(modes.shape[1]), atol=1e-10)
    assert np.all(np.diff(gen.eigenvalues) <= 1e-9)


def test_kl_fried_parameter_scales_eigenvalues_only(grid):
    a = KLBasisGenerator(grid, fried_parameter=0.1)
    b = KLBasisGenerator(grid, fried_parameter=0.2)
    ma = a.generate(10)
    b.generate(10)
    assert np.allclose(a.eigenvalues / b.eigenvalues, 2 ** (5 / 3))
    # Degenerate pairs may rotate, so check that a's modes are eigenvectors of b's covariance.
    cov_b = b._von_karman_covariance()
    assert np.allclose(cov_b @ ma, ma * b.eigenvalues, rtol=1e-8, atol=1e-8 * b.eigenvalues[0])


def test_load_keeps_basis_type_and_checks_class(grid, tmp_path):
    gen = HadamardBasisGenerator(grid)
    gen.generate(4)
    path = tmp_path / "basis.npz"
    gen.save(path)

    loaded = BasisGenerator.load(path)
    assert isinstance(loaded, ConcreteBasis)
    assert loaded.basis_type == "HadamardBasisGenerator"
    assert HadamardBasisGenerator.load(path).basis_type == "HadamardBasisGenerator"
    with pytest.raises(ValueError, match="HadamardBasisGenerator"):
        KLBasisGenerator.load(path)

    # Re-saving a loaded basis keeps the original type.
    loaded.save(tmp_path / "again.npz")
    assert BasisGenerator.load(tmp_path / "again.npz").basis_type == "HadamardBasisGenerator"


def test_numerical_rank_matches_matrix_rank():
    from aobasis.base import _numerical_rank

    rng = np.random.default_rng(0)
    a = rng.standard_normal((200, 50))
    dependent = np.hstack([a, a[:, :5] @ rng.standard_normal((5, 10))])
    nearly = np.hstack([a, a[:, :1] + 1e-10 * rng.standard_normal((200, 1))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        zernike = ZernikeBasisGenerator(
            make_circular_actuator_grid(10.0, 12), pupil_radius=5.0
        ).generate(88)  # rank 79: high orders alias on the grid
    for m in (a, dependent, nearly, zernike, np.zeros((5, 3)), np.ones((5, 3))):
        assert _numerical_rank(m) == np.linalg.matrix_rank(m)


def test_orthonormalize_modes_handles_empty():
    assert orthonormalize_modes(np.zeros((5, 0))).shape == (5, 0)


def test_positions_from_mask():
    mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    positions = positions_from_mask(mask, pitch=0.5)
    assert positions.shape == (5, 2)
    assert np.allclose(positions[0], [0.0, -0.5])  # row 0, col 1
    assert np.allclose(positions[2], [0.0, 0.0])
    assert np.allclose(positions.mean(axis=0), 0.0)
    with pytest.raises(ValueError):
        positions_from_mask(np.ones(3, bool), pitch=1.0)
    with pytest.raises(ValueError):
        positions_from_mask(mask, pitch=0.0)


def test_importing_aobasis_does_not_import_matplotlib():
    import subprocess
    import sys

    code = "import aobasis, sys; print('matplotlib.pyplot' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_importing_aobasis_does_not_import_cupy():
    import subprocess
    import sys

    code = "import aobasis, sys; print('cupy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_plotting_without_matplotlib_names_the_extra(monkeypatch):
    import sys

    from aobasis import plot_basis_modes

    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match=r"aobasis\[plot\]"):
        plot_basis_modes(np.eye(3), np.zeros((3, 2)), count=1)


@pytest.mark.parametrize(
    "make, call",
    [
        (lambda p: ZernikeBasisGenerator(p, pupil_radius=5.0), {"n_modes": 5, "orthonormalise": True}),
        (lambda p: FourierBasisGenerator(p, pupil_diameter=10.0), {"n_modes": 5, "ignore_pistons": True}),
        (lambda p: HadamardBasisGenerator(p), {"n_modes": 5, "normalise": "rms"}),
        (lambda p: KLBasisGenerator(p), {"n_modes": 5, "use_gpu": True}),
        (lambda p: ZonalBasisGenerator(p), {"n_modes": 5, "ignore_piston": True}),
        (lambda p: ZonalFastBasisGenerator(p, 1.0), {"orthonormalize": True}),
    ],
)
def test_generate_rejects_unknown_keywords(grid, make, call):
    with pytest.raises(TypeError):
        make(grid).generate(**call)


def test_concrete_basis_generate_stores_modes_and_can_grow_again(grid, tmp_path):
    gen = ZernikeBasisGenerator(grid, pupil_radius=5.0)
    gen.generate(10)
    gen.save(tmp_path / "zern")  # np.savez appends .npz
    loaded = BasisGenerator.load(tmp_path / "zern")
    assert loaded.generate(3).shape == (grid.shape[0], 3)
    assert loaded.modes.shape == (grid.shape[0], 3)
    assert np.allclose(loaded.generate(10), gen.modes)
    with pytest.raises(ValueError):
        loaded.generate(11)
