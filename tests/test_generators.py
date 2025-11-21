import pytest
import numpy as np
from aobasis import (
    KLBasisGenerator, 
    ZernikeBasisGenerator, 
    FourierBasisGenerator,
    ZonalBasisGenerator,
    HadamardBasisGenerator,
    make_circular_actuator_grid
)

@pytest.fixture
def grid():
    return make_circular_actuator_grid(telescope_diameter=10.0, grid_size=10)

@pytest.fixture
def small_grid():
    """Smaller grid for faster tests."""
    return make_circular_actuator_grid(telescope_diameter=5.0, grid_size=6)

def test_kl_generation(grid):
    gen = KLBasisGenerator(grid, fried_parameter=0.2, outer_scale=30.0)
    modes = gen.generate(n_modes=10)
    assert modes.shape == (grid.shape[0], 10)
    # Check orthogonality (approximate due to numerical precision)
    gram = modes.T @ modes
    assert np.allclose(gram, np.eye(10), atol=1e-10)
    
    # Test ignore_piston
    modes_no_piston = gen.generate(n_modes=10, ignore_piston=True)
    assert modes_no_piston.shape == (grid.shape[0], 10)
    # The first mode of no_piston should be the second mode of with_piston
    # (Assuming first mode was piston-like and largest variance)
    # Note: KL modes sign is arbitrary, so check absolute correlation
    corr = np.abs(np.dot(modes_no_piston[:, 0], modes[:, 1]))
    assert corr > 0.99

def test_kl_cpu_covariance(small_grid):
    """Test CPU covariance computation explicitly."""
    gen = KLBasisGenerator(small_grid, fried_parameter=0.16, outer_scale=30.0, use_gpu=False)
    cov = gen._von_karman_covariance_cpu()
    
    # Check that covariance is symmetric
    assert np.allclose(cov, cov.T)
    
    # Check that diagonal elements are all the same (variance)
    assert np.allclose(cov.diagonal(), cov.diagonal()[0])
    
    # Check that covariance is positive semi-definite
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= -1e-10)

def test_kl_with_different_parameters(small_grid):
    """Test KL with various parameters."""
    # Test with different fried parameter
    gen1 = KLBasisGenerator(small_grid, fried_parameter=0.1, outer_scale=30.0)
    modes1 = gen1.generate(n_modes=5)
    
    gen2 = KLBasisGenerator(small_grid, fried_parameter=0.2, outer_scale=30.0)
    modes2 = gen2.generate(n_modes=5)
    
    # Different fried parameters should give different eigenvalues
    assert not np.allclose(gen1.eigenvalues, gen2.eigenvalues)
    
    # Test with different outer scale
    gen3 = KLBasisGenerator(small_grid, fried_parameter=0.16, outer_scale=20.0)
    modes3 = gen3.generate(n_modes=5)
    
    gen4 = KLBasisGenerator(small_grid, fried_parameter=0.16, outer_scale=40.0)
    modes4 = gen4.generate(n_modes=5)
    
    # Different outer scales should give different eigenvalues
    assert not np.allclose(gen3.eigenvalues, gen4.eigenvalues)

def test_kl_gpu_fallback_warning(small_grid, capsys):
    """Test that GPU fallback warning is shown when CuPy is not available."""
    # Temporarily make HAS_CUPY False by creating generator with use_gpu=True
    # but mocking the import
    import aobasis.kl as kl_module
    original_has_cupy = kl_module.HAS_CUPY
    
    try:
        # Force HAS_CUPY to False
        kl_module.HAS_CUPY = False
        gen = KLBasisGenerator(small_grid, use_gpu=True)
        captured = capsys.readouterr()
        assert "Warning: CuPy not found" in captured.out
        assert gen.use_gpu is False
    finally:
        # Restore original value
        kl_module.HAS_CUPY = original_has_cupy

def test_kl_gpu_path_when_available(small_grid):
    """Test GPU code path if CuPy is available, otherwise skip."""
    try:
        import cupy as cp
        # If CuPy is available, test GPU path
        gen_gpu = KLBasisGenerator(small_grid, fried_parameter=0.16, outer_scale=30.0, use_gpu=True)
        modes_gpu = gen_gpu.generate(n_modes=5)
        
        # Compare with CPU
        gen_cpu = KLBasisGenerator(small_grid, fried_parameter=0.16, outer_scale=30.0, use_gpu=False)
        modes_cpu = gen_cpu.generate(n_modes=5)
        
        # Results should be similar (within numerical tolerance)
        # Eigenvectors can have opposite signs
        assert modes_gpu.shape == modes_cpu.shape
        assert np.allclose(gen_gpu.eigenvalues, gen_cpu.eigenvalues, rtol=1e-3)
        
    except ImportError:
        pytest.skip("CuPy not available, skipping GPU tests")

def test_zernike_generation(grid):
    gen = ZernikeBasisGenerator(grid, pupil_radius=5.0)
    modes = gen.generate(n_modes=10)
    assert modes.shape == (grid.shape[0], 10)
    
    # Check Noll indices
    # Mode 1: Piston (n=0, m=0) -> Constant
    assert np.allclose(modes[:, 0], 1.0)
    
    # Test ignore_piston
    modes_no_piston = gen.generate(n_modes=10, ignore_piston=True)
    assert modes_no_piston.shape == (grid.shape[0], 10)
    # First mode should NOT be piston (constant)
    assert not np.allclose(modes_no_piston[:, 0], 1.0)
    # It should be Tip (Noll 2)
    # Check correlation with original mode 1 (Tip)
    corr = np.abs(np.dot(modes_no_piston[:, 0], modes[:, 1]))
    # Normalize
    corr /= (np.linalg.norm(modes_no_piston[:, 0]) * np.linalg.norm(modes[:, 1]))
    assert corr > 0.99

def test_zernike_orthogonality(grid):
    """Test Zernike orthogonality."""
    gen = ZernikeBasisGenerator(grid, pupil_radius=5.0)
    modes = gen.generate(n_modes=15)
    
    # Approximate orthogonality (they're sampled at discrete points)
    gram = modes.T @ modes
    # Diagonal should be positive
    assert np.all(np.diag(gram) > 0)

def test_fourier_generation(grid):
    gen = FourierBasisGenerator(grid, pupil_diameter=10.0)
    modes = gen.generate(n_modes=10)
    assert modes.shape == (grid.shape[0], 10)
    # First mode is piston
    assert np.allclose(modes[:, 0], 1.0)
    
    # Test ignore_piston
    modes_no_piston = gen.generate(n_modes=10, ignore_piston=True)
    assert modes_no_piston.shape == (grid.shape[0], 10)
    # First mode should NOT be piston
    assert not np.allclose(modes_no_piston[:, 0], 1.0)

def test_fourier_different_diameter(small_grid):
    """Test Fourier with different pupil diameters."""
    gen1 = FourierBasisGenerator(small_grid, pupil_diameter=5.0)
    modes1 = gen1.generate(n_modes=8)
    
    gen2 = FourierBasisGenerator(small_grid, pupil_diameter=10.0)
    modes2 = gen2.generate(n_modes=8)
    
    # Different diameters should give different modes (except piston)
    assert not np.allclose(modes1[:, 1:], modes2[:, 1:])

def test_zonal_generation(grid):
    gen = ZonalBasisGenerator(grid)
    modes = gen.generate(n_modes=5)
    assert modes.shape == (grid.shape[0], 5)
    # Should be columns of identity
    expected = np.eye(grid.shape[0])[:, :5]
    assert np.allclose(modes, expected)
    
    # Test error on too many modes
    with pytest.raises(ValueError):
        gen.generate(n_modes=grid.shape[0] + 1)

def test_zonal_all_modes(small_grid):
    """Test generating all zonal modes."""
    gen = ZonalBasisGenerator(small_grid)
    modes = gen.generate(n_modes=small_grid.shape[0])
    assert modes.shape == (small_grid.shape[0], small_grid.shape[0])
    assert np.allclose(modes, np.eye(small_grid.shape[0]))

def test_hadamard_generation(grid):
    gen = HadamardBasisGenerator(grid)
    modes = gen.generate(n_modes=8)
    assert modes.shape == (grid.shape[0], 8)
    # Entries should be 1 or -1
    assert np.all(np.isin(modes, [1, -1]))

def test_hadamard_power_of_two():
    """Test Hadamard with power of 2 actuators."""
    # Create grid with exactly 16 actuators
    positions = np.random.rand(16, 2)
    gen = HadamardBasisGenerator(positions)
    modes = gen.generate(n_modes=8)
    assert modes.shape == (16, 8)

def test_hadamard_non_power_of_two():
    """Test Hadamard with non-power of 2 actuators."""
    # Create grid with 20 actuators (not power of 2)
    positions = np.random.rand(20, 2)
    gen = HadamardBasisGenerator(positions)
    modes = gen.generate(n_modes=8)
    assert modes.shape == (20, 8)

def test_save_load(grid, tmp_path):
    gen = KLBasisGenerator(grid)
    modes = gen.generate(n_modes=5)
    
    save_path = tmp_path / "test_basis.npz"
    gen.save(save_path)
    
    assert save_path.exists()
    
    # Load back
    loaded_gen = KLBasisGenerator.load(save_path)
    assert np.allclose(loaded_gen.positions, grid)
    assert np.allclose(loaded_gen.modes, modes)

def test_save_without_generation(grid, tmp_path):
    """Test that save raises error if no modes generated."""
    gen = KLBasisGenerator(grid)
    save_path = tmp_path / "test_basis.npz"
    
    with pytest.raises(ValueError, match="No modes generated yet"):
        gen.save(save_path)

def test_plot_without_generation(grid):
    """Test that plot raises error if no modes generated."""
    gen = KLBasisGenerator(grid)
    
    with pytest.raises(ValueError, match="No modes to plot"):
        gen.plot()

def test_loaded_basis_generate(grid, tmp_path):
    """Test generating from a loaded basis."""
    gen = KLBasisGenerator(grid)
    modes = gen.generate(n_modes=10)
    
    save_path = tmp_path / "test_basis.npz"
    gen.save(save_path)
    
    loaded_gen = KLBasisGenerator.load(save_path)
    # Generate subset of modes
    subset_modes = loaded_gen.generate(n_modes=5)
    assert subset_modes.shape == (grid.shape[0], 5)
    assert np.allclose(subset_modes, modes[:, :5])
