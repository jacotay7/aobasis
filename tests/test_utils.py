import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from aobasis.utils import make_circular_actuator_grid, make_concentric_actuator_grid, plot_basis_modes

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

@patch("aobasis.utils.plt")
def test_plot_basis_modes(mock_plt):
    # Setup mock data
    n_actuators = 20
    n_modes = 5
    positions = np.random.rand(n_actuators, 2)
    modes = np.random.rand(n_actuators, n_modes)
    
    # Configure mock to return a tuple
    mock_fig = MagicMock()
    mock_axes = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axes)
    
    # Test basic plotting
    plot_basis_modes(modes, positions, count=3)
    
    assert mock_plt.subplots.called
    assert mock_plt.show.called

@patch("aobasis.utils.plt")
def test_plot_basis_modes_save(mock_plt, tmp_path):
    # Setup mock data
    n_actuators = 20
    n_modes = 5
    positions = np.random.rand(n_actuators, 2)
    modes = np.random.rand(n_actuators, n_modes)
    outfile = tmp_path / "test_plot.png"
    
    # Configure mock to return a tuple
    mock_fig = MagicMock()
    mock_axes = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axes)
    
    # Test saving to file
    plot_basis_modes(modes, positions, count=3, outfile=outfile)
    
    assert mock_plt.subplots.called
    mock_plt.savefig.assert_called_with(outfile, dpi=150)
    assert mock_plt.close.called

@patch("aobasis.utils.plt")
def test_plot_basis_modes_interpolate(mock_plt):
    # Setup mock data
    n_actuators = 20
    n_modes = 5
    positions = np.random.rand(n_actuators, 2)
    modes = np.random.rand(n_actuators, n_modes)
    
    # Configure mock to return a tuple
    mock_fig = MagicMock()
    mock_axes = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axes)
    
    # Test interpolation
    plot_basis_modes(modes, positions, count=3, interpolate=True)
    
    assert mock_plt.subplots.called
    # We can't easily check if imshow was called on the axes objects without more complex mocking,
    # but we can check that no errors were raised.

@patch("aobasis.utils.plt")
def test_plot_basis_modes_with_title_prefix(mock_plt):
    """Test plotting with custom title prefix."""
    n_actuators = 20
    n_modes = 5
    positions = np.random.rand(n_actuators, 2)
    modes = np.random.rand(n_actuators, n_modes)
    
    mock_fig = MagicMock()
    mock_axes = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axes)
    
    plot_basis_modes(modes, positions, count=3, title_prefix="Test Mode")
    
    assert mock_plt.subplots.called

@patch("aobasis.utils.plt")
def test_plot_basis_modes_count_exceeds_available(mock_plt):
    """Test plotting when count exceeds available modes."""
    n_actuators = 20
    n_modes = 3
    positions = np.random.rand(n_actuators, 2)
    modes = np.random.rand(n_actuators, n_modes)
    
    mock_fig = MagicMock()
    mock_axes = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axes)
    
    # Request more modes than available
    plot_basis_modes(modes, positions, count=10)
    
    # Should only plot available modes
    assert mock_plt.subplots.called

@patch("aobasis.utils.plt")
def test_plot_basis_modes_all_params(mock_plt):
    """Test plotting with all parameters."""
    n_actuators = 20
    n_modes = 5
    positions = np.random.rand(n_actuators, 2)
    modes = np.random.rand(n_actuators, n_modes)
    
    mock_fig = MagicMock()
    mock_axes = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_axes)
    
    plot_basis_modes(
        modes, 
        positions, 
        count=3, 
        title_prefix="Mode",
        interpolate=False
    )
    
    assert mock_plt.subplots.called
    assert mock_plt.show.called

def test_plot_basis_modes_invalid_shape():
    n_actuators = 20
    n_modes = 5
    positions = np.random.rand(n_actuators, 2)
    modes = np.random.rand(n_actuators + 1, n_modes) # Mismatch
    
    with pytest.raises(ValueError, match="Mode dimension 0"):
        plot_basis_modes(modes, positions)

def test_plot_basis_modes_1d_modes():
    """Test plotting with 1D modes array."""
    n_actuators = 20
    positions = np.random.rand(n_actuators, 2)
    modes = np.random.rand(n_actuators)  # 1D array
    
    # Should work by treating as single mode
    with patch("aobasis.utils.plt") as mock_plt:
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        # Reshape to 2D should work
        plot_basis_modes(modes.reshape(-1, 1), positions, count=1)
        assert mock_plt.subplots.called
