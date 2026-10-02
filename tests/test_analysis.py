"""Coefficient fitting, C2M and the basis report (#35)."""

import numpy as np
import pytest

from aobasis import (
    HadamardBasisGenerator,
    KLBasisGenerator,
    ZernikeBasisGenerator,
    basis_report,
    command_to_mode_matrix,
    fit_coefficients,
    make_circular_actuator_grid,
)


@pytest.fixture
def grid():
    return make_circular_actuator_grid(telescope_diameter=10.0, grid_size=12)


def test_fit_coefficients_recovers_modal_content(grid):
    modes = ZernikeBasisGenerator(grid, pupil_radius=5.0).generate(20)
    true = np.random.default_rng(0).standard_normal((20, 4))
    commands = modes @ true
    assert np.allclose(fit_coefficients(modes, commands), true)
    assert np.allclose(fit_coefficients(modes, commands[:, 0]), true[:, 0])
    c2m = command_to_mode_matrix(modes)
    assert c2m.shape == (20, grid.shape[0])
    assert np.allclose(c2m @ commands, true)
    damped = fit_coefficients(modes, commands, regularization=1.0)
    assert np.linalg.norm(damped) < np.linalg.norm(true)
    with pytest.raises(ValueError):
        fit_coefficients(modes, commands[:-1])


def test_fit_coefficients_projects_out_of_span_commands(grid):
    modes = KLBasisGenerator(grid).generate(10)  # orthonormal
    commands = np.random.default_rng(1).standard_normal(grid.shape[0])
    assert np.allclose(fit_coefficients(modes, commands), modes.T @ commands)


def test_basis_report(grid):
    kl = KLBasisGenerator(grid)
    kl.generate(15, ignore_piston=True)
    report = kl.report()
    assert report.rank == 15 and report.n_actuators == grid.shape[0]
    assert report.condition_number == pytest.approx(1.0)
    assert report.max_cosine < 1e-10
    assert report.piston_content.max() < 1e-10
    assert "15 modes" in str(report)

    hadamard = basis_report(HadamardBasisGenerator(grid).generate(10))
    assert hadamard.piston_content[0] == pytest.approx(1.0)  # column 0 is piston
    dependent = basis_report(np.column_stack([np.ones(5), np.ones(5)]))
    assert dependent.rank == 1 and dependent.condition_number == float("inf")
    assert dependent.max_cosine == pytest.approx(1.0)
    with pytest.raises(ValueError):
        KLBasisGenerator(grid).report()  # nothing generated yet
