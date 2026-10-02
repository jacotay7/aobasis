import pytest
import numpy as np
from aobasis.utils import make_circular_actuator_grid, make_concentric_actuator_grid

def test_make_circular_actuator_grid():
    diameter = 10.0
    grid_size = 10
    positions = make_circular_actuator_grid(diameter, grid_size)
    
    assert isinstance(positions, np.ndarray)
    assert positions.shape[1] == 2
    
    # Check that all points are within the radius
    radius = diameter / 2
    distances = np.linalg.norm(positions, axis=1)
    assert np.all(distances <= radius * 1.0000001)

def test_make_circular_actuator_grid_different_sizes():
    """Test circular grid with different sizes."""
    # Small grid
    positions_small = make_circular_actuator_grid(5.0, 5)
    assert positions_small.shape[0] > 0
    
    # Large grid
    positions_large = make_circular_actuator_grid(20.0, 20)
    assert positions_large.shape[0] > positions_small.shape[0]

def test_make_circular_actuator_grid_center_included():
    """Test that grid includes points near center."""
    positions = make_circular_actuator_grid(10.0, 10)
    # Check if there are points within inner region
    distances = np.linalg.norm(positions, axis=1)
    # For a circular grid, innermost points should be within some radius
    # The actual implementation may or may not include exact center
    assert np.any(distances < 2.0)  # At least some points in inner region

def test_make_concentric_actuator_grid():
    diameter = 10.0
    n_rings = 3
    n_points_innermost = 6
    positions = make_concentric_actuator_grid(diameter, n_rings, n_points_innermost)
    
    assert isinstance(positions, np.ndarray)
    assert positions.shape[1] == 2
    
    # Expected number of points: 1 (center) + 6*1 + 6*2 + 6*3 = 1 + 6 + 12 + 18 = 37
    expected_points = 1 + sum(n_points_innermost * i for i in range(1, n_rings + 1))
    assert positions.shape[0] == expected_points

def test_make_concentric_actuator_grid_single_ring():
    """Test concentric grid with single ring."""
    positions = make_concentric_actuator_grid(10.0, n_rings=1, n_points_innermost=8)
    # Should have center + first ring
    expected_points = 1 + 8
    assert positions.shape[0] == expected_points

def test_make_concentric_actuator_grid_center():
    """Test that center point is at origin."""
    positions = make_concentric_actuator_grid(10.0, n_rings=2, n_points_innermost=6)
    # First point should be at origin
    assert np.allclose(positions[0], [0, 0])

def test_make_concentric_actuator_grid_radius():
    """Test that all points are within radius."""
    diameter = 10.0
    positions = make_concentric_actuator_grid(diameter, n_rings=3, n_points_innermost=6)
    radius = diameter / 2
    distances = np.linalg.norm(positions, axis=1)
    assert np.all(distances <= radius * 1.0000001)


def test_make_circular_actuator_grid_small_sizes():
    assert np.array_equal(make_circular_actuator_grid(10.0, 1), np.zeros((1, 2)))
    with pytest.raises(ValueError, match="grid_size=2"):
        make_circular_actuator_grid(10.0, 2)
    three = make_circular_actuator_grid(10.0, 3)
    assert three.shape == (5, 2)  # centre and the four rim points on the axes


def test_make_circular_actuator_grid_pitch():
    positions = make_circular_actuator_grid(10.0, 21)
    xs = np.unique(np.round(positions[:, 0], 12))
    assert np.allclose(np.diff(xs), 10.0 / 20)
    assert np.isclose(np.abs(positions).max(), 5.0)
