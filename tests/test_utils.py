import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
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

def test_plot_basis_modes_invalid_shape():
    n_actuators = 20
    n_modes = 5
    positions = np.random.rand(n_actuators, 2)
    modes = np.random.rand(n_actuators + 1, n_modes) # Mismatch
    
    with pytest.raises(ValueError, match="Mode dimension 0"):
        plot_basis_modes(modes, positions)
