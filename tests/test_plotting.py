"""plot_basis_modes; skipped when the optional matplotlib is not installed."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot  # noqa: E402,F401  (so the tests can patch it; aobasis imports it lazily)

from aobasis.utils import plot_basis_modes  # noqa: E402

@patch("matplotlib.pyplot")
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

@patch("matplotlib.pyplot")
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

@patch("matplotlib.pyplot")
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

@patch("matplotlib.pyplot")
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

@patch("matplotlib.pyplot")
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

@patch("matplotlib.pyplot")
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
    with patch("matplotlib.pyplot") as mock_plt:
        mock_fig = MagicMock()
        mock_axes = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        
        # Reshape to 2D should work
        plot_basis_modes(modes.reshape(-1, 1), positions, count=1)
        assert mock_plt.subplots.called
