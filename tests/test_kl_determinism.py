"""KL modes follow a fixed sign and degenerate-rotation convention (#20)."""

import numpy as np
import pytest
from scipy.linalg import eigh as scipy_eigh

import aobasis.kl as kl_module
from aobasis import KLBasisGenerator, make_circular_actuator_grid


@pytest.fixture
def grid():
    return make_circular_actuator_grid(telescope_diameter=10.0, grid_size=16)


def _scrambling_eigh(seed):
    """eigh that rotates degenerate eigenvectors and flips signs at random."""
    rng = np.random.default_rng(seed)

    def eigh(cov, **kwargs):
        values, vectors = scipy_eigh(cov, **kwargs)
        vectors = vectors * rng.choice([-1.0, 1.0], size=vectors.shape[1])
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and abs(values[stop] - values[start]) <= 1e-10 * abs(values[start]):
                stop += 1
            if stop - start > 1:
                rot, _ = np.linalg.qr(rng.standard_normal((stop - start, stop - start)))
                vectors[:, start:stop] = vectors[:, start:stop] @ rot
            start = stop
        return values, vectors

    return eigh


def test_grid_has_degenerate_kl_pairs(grid):
    gen = KLBasisGenerator(grid)
    gen.generate(30)
    ratios = gen.eigenvalues[1:] / gen.eigenvalues[:-1]
    assert np.sum(ratios > 1 - 1e-10) >= 5


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_modes_do_not_depend_on_eigensolver_rotation_or_sign(grid, monkeypatch, seed):
    reference = KLBasisGenerator(grid).generate(40, ignore_piston=True)
    monkeypatch.setattr(kl_module, "eigh", _scrambling_eigh(seed))
    scrambled = KLBasisGenerator(grid).generate(40, ignore_piston=True)
    assert np.allclose(scrambled, reference, atol=1e-10)


def test_cutting_a_degenerate_pair_keeps_the_convention(grid):
    gen = KLBasisGenerator(grid)
    gen.generate(30)
    values = gen.eigenvalues
    cut = next(k for k in range(1, 30) if abs(values[k] - values[k - 1]) <= 1e-8 * values[k - 1])
    assert np.allclose(KLBasisGenerator(grid).generate(cut), gen.modes[:, :cut], atol=1e-10)


def test_permuting_actuators_permutes_the_modes(grid):
    perm = np.random.default_rng(3).permutation(grid.shape[0])
    modes = KLBasisGenerator(grid).generate(25)
    permuted = KLBasisGenerator(grid[perm]).generate(25)
    assert np.allclose(permuted, modes[perm], atol=1e-10)


def test_modes_stay_orthonormal_eigenvectors(grid):
    gen = KLBasisGenerator(grid)
    modes = gen.generate(40)
    cov = gen._von_karman_covariance_cpu()
    assert np.allclose(modes.T @ modes, np.eye(40), atol=1e-12)
    assert np.allclose(modes.T @ cov @ modes, np.diag(gen.eigenvalues), atol=1e-9 * gen.eigenvalues[0])


def test_gpu_modes_match_cpu(grid, gpu):
    cpu = KLBasisGenerator(grid).generate(40, ignore_piston=True)
    on_gpu = KLBasisGenerator(grid, use_gpu=True).generate(40, ignore_piston=True)
    assert np.allclose(on_gpu, cpu, atol=1e-8)
