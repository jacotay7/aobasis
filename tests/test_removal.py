"""ignore_piston and remove= keep the given modes out of every basis (#14, #27)."""

import numpy as np
import pytest

from aobasis import (
    FourierBasisGenerator,
    HadamardBasisGenerator,
    KLBasisGenerator,
    ZernikeBasisGenerator,
    make_circular_actuator_grid,
    project_out,
    removal_basis,
)


@pytest.fixture
def grid():
    return make_circular_actuator_grid(telescope_diameter=10.0, grid_size=12)


GENERATORS = {
    "zernike": lambda p: ZernikeBasisGenerator(p, pupil_radius=5.0),
    "fourier": lambda p: FourierBasisGenerator(p, pupil_diameter=10.0),
    "hadamard": lambda p: HadamardBasisGenerator(p),
    "kl": lambda p: KLBasisGenerator(p),
}


@pytest.mark.parametrize("name", GENERATORS)
@pytest.mark.parametrize("orthonormalize", [False, True])
def test_ignore_piston_gives_zero_mean_modes(grid, name, orthonormalize):
    modes = GENERATORS[name](grid).generate(20, ignore_piston=True, orthonormalize=orthonormalize)
    assert modes.shape == (grid.shape[0], 20)
    rms = np.sqrt(np.mean(modes**2, axis=0))
    assert np.all(np.abs(modes.mean(axis=0)) <= 1e-12 * rms)
    if orthonormalize:
        assert np.allclose(modes.T @ modes, np.eye(20), atol=1e-10)


@pytest.mark.parametrize("name", GENERATORS)
def test_remove_tiptilt_and_arrays(grid, name):
    rng = np.random.default_rng(1)
    extra = rng.standard_normal(grid.shape[0])
    modes = GENERATORS[name](grid).generate(
        15, ignore_piston=True, orthonormalize=True, remove=["tiptilt", extra]
    )
    blocked = np.column_stack([np.ones(grid.shape[0]), grid, extra])
    assert np.abs(blocked.T @ modes).max() <= 1e-10 * np.linalg.norm(blocked, axis=0).max()
    assert np.linalg.matrix_rank(modes) == 15


def test_zernike_skips_modes_inside_the_removed_subspace(grid):
    gen = ZernikeBasisGenerator(grid, pupil_radius=5.0)
    plain = gen.generate(6)
    without = gen.generate(3, ignore_piston=True, remove="tiptilt")
    # Piston, tip and tilt are dropped, so the first mode is defocus (j=4).
    for k, j in enumerate((4, 5, 6)):
        ref = plain[:, j - 1] - plain[:, j - 1].mean()
        corr = abs(ref @ without[:, k]) / (np.linalg.norm(ref) * np.linalg.norm(without[:, k]))
        assert corr > 0.999


def test_hadamard_ignore_piston_still_skips_column_zero(grid):
    modes = HadamardBasisGenerator(grid).generate(5, ignore_piston=True)
    from scipy.linalg import hadamard

    H = hadamard(128)[: grid.shape[0]].astype(float)
    assert np.allclose(modes, H[:, 1:6] - H[:, 1:6].mean(axis=0))


def test_kl_remove_tiptilt_matches_projected_covariance(grid):
    gen = KLBasisGenerator(grid)
    modes = gen.generate(10, ignore_piston=True, remove="tiptilt")
    u = removal_basis(grid, "tiptilt", ignore_piston=True)
    p = np.eye(grid.shape[0]) - u @ u.T
    cov = p @ gen._von_karman_covariance_cpu() @ p
    assert np.allclose(modes.T @ modes, np.eye(10), atol=1e-10)
    assert np.allclose(modes.T @ cov @ modes, np.diag(gen.eigenvalues), atol=1e-8 * gen.eigenvalues[0])
    assert np.all(np.diff(gen.eigenvalues) <= 0)


@pytest.mark.parametrize("name", GENERATORS)
def test_mode_limit_accounts_for_removed_modes(grid, name):
    n = grid.shape[0]
    gen = GENERATORS[name](grid)
    with pytest.raises(ValueError):
        gen.generate(n - 2, ignore_piston=True, remove="tiptilt")


def test_removal_basis_and_project_out():
    positions = make_circular_actuator_grid(10.0, 8)
    n = positions.shape[0]
    u = removal_basis(positions, ["piston", "tip", "tiptilt"])  # tip twice: rank 3
    assert u.shape == (n, 3)
    assert np.allclose(u.T @ u, np.eye(3))
    assert removal_basis(positions).shape == (n, 0)

    rng = np.random.default_rng(0)
    modes = rng.standard_normal((n, 4))
    sub = rng.standard_normal((n, 2))
    out = project_out(modes, sub)
    assert np.allclose(sub.T @ out, 0, atol=1e-10)
    assert np.allclose(project_out(out, sub), out)
    assert np.allclose(project_out(modes, sub[:, 0]), project_out(modes, sub[:, :1]))


@pytest.mark.parametrize(
    "bad",
    ["defocus", np.ones(3), np.ones((3, 2)), [1.0, 2.0], np.array([np.nan] * 32), 5],
)
def test_removal_basis_rejects_bad_input(bad):
    positions = make_circular_actuator_grid(10.0, 8)  # 32 actuators
    with pytest.raises(ValueError):
        removal_basis(positions, bad)


def test_kl_remove_on_gpu_matches_cpu(grid, gpu):
    cpu = KLBasisGenerator(grid)
    cpu.generate(10, ignore_piston=True, remove="tiptilt")
    gpu_gen = KLBasisGenerator(grid, use_gpu=True)
    gpu_gen.generate(10, ignore_piston=True, remove="tiptilt")
    assert np.allclose(gpu_gen.eigenvalues, cpu.eigenvalues, rtol=1e-9)
