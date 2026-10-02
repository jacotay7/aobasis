"""KL: distinct-distance covariance, partial eigensolver, Kolmogorov, wavelength (#28)."""

import numpy as np
import pytest
from scipy.linalg import subspace_angles
from scipy.special import gamma, kv

import aobasis.kl as kl_module
from aobasis import KLBasisGenerator, make_circular_actuator_grid, removal_basis


@pytest.fixture
def grid():
    return make_circular_actuator_grid(telescope_diameter=10.0, grid_size=30)


def test_covariance_matches_direct_formula(grid):
    gen = KLBasisGenerator(grid, fried_parameter=0.12, outer_scale=25.0)
    r = np.linalg.norm(grid[:, None] - grid[None], axis=-1)
    a = (5 / 6) * (6.88 / 2) * gamma(5 / 6) / (gamma(1 / 6) * np.pi ** (5 / 3))
    sigma2 = a * (25.0 / 0.12) ** (5 / 3)
    u = 2 * np.pi * np.where(r > 0, r, 1.0) / 25.0
    direct = np.where(r > 0, sigma2 * 2 ** (1 / 6) / gamma(5 / 6) * u ** (5 / 6) * kv(5 / 6, u), sigma2)
    assert np.allclose(gen._von_karman_covariance_cpu(), direct, rtol=1e-13, atol=0)


def test_partial_eigensolver_matches_full(grid, monkeypatch):
    partial = KLBasisGenerator(grid)
    partial.generate(25, ignore_piston=True)
    monkeypatch.setattr(kl_module, "_SUBSET_FRACTION", 0.0)  # always the full solver
    full = KLBasisGenerator(grid)
    full.generate(25, ignore_piston=True)
    assert np.allclose(partial.eigenvalues, full.eigenvalues, rtol=1e-10)
    assert np.allclose(partial.modes, full.modes, atol=1e-9)


@pytest.mark.parametrize("remove, noll", [(None, 1.0299), ("tiptilt", 0.134)])
def test_kolmogorov_residual_variance_matches_noll(grid, remove, noll):
    # Noll (1976): aperture-averaged phase variance with piston (Delta_1) or
    # piston and tip/tilt (Delta_3) removed, in units of (D/r0)^(5/3).
    gen = KLBasisGenerator(grid, fried_parameter=0.16, outer_scale=np.inf)
    u = removal_basis(grid, remove, ignore_piston=True)
    p = np.eye(grid.shape[0]) - u @ u.T
    total = np.trace(p @ gen._von_karman_covariance_cpu() @ p) / grid.shape[0]
    assert total / (10.0 / 0.16) ** (5 / 3) == pytest.approx(noll, rel=0.03)


def test_kolmogorov_needs_piston_removed(grid):
    gen = KLBasisGenerator(grid, outer_scale=np.inf)
    with pytest.raises(ValueError, match="ignore_piston"):
        gen.generate(10)
    modes = gen.generate(10, ignore_piston=True)
    assert np.allclose(modes.mean(axis=0), 0, atol=1e-12)
    assert np.all(np.diff(gen.eigenvalues) <= 0) and np.all(gen.eigenvalues > 0)


def test_large_outer_scale_approaches_kolmogorov(grid):
    kolmogorov = KLBasisGenerator(grid, outer_scale=np.inf).generate(20, ignore_piston=True)
    von_karman = KLBasisGenerator(grid, outer_scale=1e5).generate(20, ignore_piston=True)
    assert np.degrees(subspace_angles(kolmogorov, von_karman).max()) < 0.2


def test_wavelength_scales_eigenvalues_only(grid):
    at_r0 = KLBasisGenerator(grid)
    modes = at_r0.generate(10)
    at_h = KLBasisGenerator(grid, r0_wavelength=500e-9, wavelength=1.65e-6)
    assert np.allclose(at_h.generate(10), modes)
    assert np.allclose(at_h.eigenvalues, at_r0.eigenvalues * (0.5 / 1.65) ** 2)
    with pytest.raises(ValueError):
        KLBasisGenerator(grid, wavelength=-1.0)


def test_kolmogorov_gpu_matches_cpu(grid, gpu):
    cpu = KLBasisGenerator(grid, outer_scale=np.inf)
    cpu.generate(20, ignore_piston=True)
    on_gpu = KLBasisGenerator(grid, outer_scale=np.inf, use_gpu=True)
    on_gpu.generate(20, ignore_piston=True)
    assert np.allclose(on_gpu.eigenvalues, cpu.eigenvalues, rtol=1e-9)
    assert np.allclose(on_gpu.modes, cpu.modes, atol=1e-8)
