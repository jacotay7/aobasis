import numpy as np
import pytest

from aobasis import (
    FourierBasisGenerator,
    HadamardBasisGenerator,
    KLBasisGenerator,
    ZernikeBasisGenerator,
    ZonalBasisGenerator,
    ZonalFastBasisGenerator,
    make_circular_actuator_grid,
    make_concentric_actuator_grid,
    plot_basis_modes,
)


def test_generators_reject_malformed_positions():
    bad_positions_list = [
        [0.0, 1.0, 2.0],
        np.array([1.0, 2.0, 3.0]),
        np.array([[0.0], [1.0]]),
        np.array([[0.0, 1.0, 2.0]]),
        np.array([[0.0, np.nan], [1.0, 2.0]]),
        np.array([[0.0, np.inf], [1.0, 2.0]]),
    ]

    generator_factories = [
        lambda positions: KLBasisGenerator(positions),
        lambda positions: ZernikeBasisGenerator(positions, pupil_radius=1.0),
        lambda positions: FourierBasisGenerator(positions, pupil_diameter=1.0),
        lambda positions: ZonalBasisGenerator(positions),
        lambda positions: ZonalFastBasisGenerator(positions, min_distance=1.0),
        lambda positions: HadamardBasisGenerator(positions),
    ]

    for bad_positions in bad_positions_list:
        for factory in generator_factories:
            with pytest.raises(ValueError):
                factory(bad_positions)


def test_geometry_helpers_reject_invalid_inputs():
    with pytest.raises(ValueError):
        make_circular_actuator_grid(0.0, 10)
    with pytest.raises(ValueError):
        make_circular_actuator_grid(10.0, 0)
    with pytest.raises(ValueError):
        make_circular_actuator_grid(np.nan, 10)

    with pytest.raises(ValueError):
        make_concentric_actuator_grid(10.0, -1)
    with pytest.raises(ValueError):
        make_concentric_actuator_grid(10.0, 2, n_points_innermost=0)
    with pytest.raises(ValueError):
        make_concentric_actuator_grid(np.inf, 2)


def test_plot_basis_modes_rejects_invalid_arguments():
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    modes = np.array([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError):
        plot_basis_modes(modes, positions[:, 0])
    with pytest.raises(ValueError):
        plot_basis_modes(modes, positions, count=-1)
    with pytest.raises(ValueError):
        plot_basis_modes(modes, positions, resolution=0, interpolate=True)
    with pytest.raises(ValueError):
        plot_basis_modes(np.array([[1.0, np.nan], [0.0, 1.0]]), positions)


def test_generators_reject_invalid_parameters():
    positions = make_circular_actuator_grid(5.0, 6)

    with pytest.raises(ValueError):
        KLBasisGenerator(positions, fried_parameter=0.0)
    with pytest.raises(ValueError):
        KLBasisGenerator(positions, outer_scale=0.0)
    with pytest.raises(ValueError):
        ZernikeBasisGenerator(positions, pupil_radius=0.0)
    with pytest.raises(ValueError):
        FourierBasisGenerator(positions, pupil_diameter=0.0)
    with pytest.raises(ValueError):
        ZonalFastBasisGenerator(positions, min_distance=np.nan)


def test_generators_handle_zero_modes_and_reject_negative_modes():
    positions = make_circular_actuator_grid(5.0, 6)

    generators = [
        KLBasisGenerator(positions),
        ZernikeBasisGenerator(positions, pupil_radius=2.5),
        FourierBasisGenerator(positions, pupil_diameter=5.0),
        ZonalBasisGenerator(positions),
        ZonalFastBasisGenerator(positions, min_distance=0.8),
        HadamardBasisGenerator(positions),
    ]

    for generator in generators:
        zero_modes = generator.generate(n_modes=0)
        assert zero_modes.shape == (positions.shape[0], 0)

        with pytest.raises(ValueError):
            generator.generate(n_modes=-1)


def test_generators_reject_requests_exceeding_available_degrees_of_freedom():
    positions = make_circular_actuator_grid(5.0, 6)
    n_actuators = positions.shape[0]

    with pytest.raises(ValueError):
        KLBasisGenerator(positions).generate(n_modes=n_actuators + 1)
    with pytest.raises(ValueError):
        KLBasisGenerator(positions).generate(n_modes=n_actuators, ignore_piston=True)
    with pytest.raises(ValueError):
        ZernikeBasisGenerator(positions, pupil_radius=2.5).generate(n_modes=-3)
    with pytest.raises(ValueError):
        FourierBasisGenerator(positions, pupil_diameter=5.0).generate(n_modes=-3)
    with pytest.raises(ValueError):
        HadamardBasisGenerator(positions).generate(n_modes=-3)
