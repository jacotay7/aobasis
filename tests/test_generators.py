import pytest
import numpy as np
from aobasis import (
    KLBasisGenerator, 
    ZernikeBasisGenerator, 
    FourierBasisGenerator,
    ZonalBasisGenerator,
    ZonalFastBasisGenerator,
    HadamardBasisGenerator,
    make_circular_actuator_grid
)


def assert_min_spacing_within_modes(positions, modes, min_distance):
    for mode_index in range(modes.shape[1]):
        active = positions[modes[:, mode_index] > 0.5]
        if active.shape[0] < 2:
            continue
        deltas = active[:, None, :] - active[None, :, :]
        distances = np.linalg.norm(deltas, axis=-1)
        upper_triangle = distances[np.triu_indices(active.shape[0], k=1)]
        assert np.all(upper_triangle >= min_distance - 1e-12)

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
    # Removing piston leaves zero-mean modes; the leading tip/tilt pair spans
    # the same space as modes 1-2 of the full basis (the pair is degenerate,
    # so individual vectors can rotate within it).
    assert np.allclose(modes_no_piston.mean(axis=0), 0.0, atol=1e-12)
    tip_tilt = modes_no_piston[:, :2]
    reference = modes[:, 1:3]
    assert np.allclose(tip_tilt @ tip_tilt.T, reference @ reference.T, atol=1e-6)

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

def test_kl_gpu_fallback_warning(small_grid, monkeypatch):
    """use_gpu=True without CuPy warns and falls back to the CPU."""
    import aobasis.kl as kl_module

    monkeypatch.setattr(kl_module, "_load_cupy", lambda: None)
    with pytest.warns(RuntimeWarning, match="CuPy not found"):
        gen = KLBasisGenerator(small_grid, use_gpu=True)
    assert gen.use_gpu is False


def test_kl_gpu_path_when_available(small_grid, gpu):
    """GPU and CPU KL agree (runs only with CuPy and a CUDA device)."""
    gen_gpu = KLBasisGenerator(small_grid, fried_parameter=0.16, outer_scale=30.0, use_gpu=True)
    modes_gpu = gen_gpu.generate(n_modes=5)
    gen_cpu = KLBasisGenerator(small_grid, fried_parameter=0.16, outer_scale=30.0, use_gpu=False)
    modes_cpu = gen_cpu.generate(n_modes=5)

    assert modes_gpu.shape == modes_cpu.shape
    assert np.allclose(gen_gpu.eigenvalues, gen_cpu.eigenvalues, rtol=1e-9)

def test_gpu_kv56_kernel_matches_scipy(gpu):
    from scipy.special import kv

    from aobasis.kl import _load_cupy

    cp, kv56 = _load_cupy()
    z = np.concatenate([np.geomspace(1e-8, 2.0, 300), np.linspace(2.0, 60.0, 300)])
    out = cp.zeros(z.size)
    kv56(cp.asarray(z), out)
    assert np.allclose(cp.asnumpy(out), kv(5.0 / 6.0, z), rtol=1e-12, atol=0)


@pytest.mark.parametrize("outer_scale", [30.0, 5.0])  # 5 m puts most pairs at 2*pi*r/L0 > 2
def test_gpu_covariance_matches_cpu(gpu, outer_scale):
    positions = make_circular_actuator_grid(telescope_diameter=10.0, grid_size=16)
    from aobasis.kl import _load_cupy

    gen = KLBasisGenerator(positions, outer_scale=outer_scale, use_gpu=True)
    cov_gpu = _load_cupy()[0].asnumpy(gen._von_karman_covariance_gpu())
    cov_cpu = gen._von_karman_covariance_cpu()
    assert np.allclose(cov_gpu, cov_cpu, rtol=1e-12, atol=0)


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

def test_zonal_fast_generation_path_graph():
    positions = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])

    gen = ZonalFastBasisGenerator(positions, min_distance=1.1)
    modes = gen.generate()

    assert modes.shape == (3, 2)
    assert np.allclose(modes.sum(axis=1), 1.0)
    assert np.allclose(modes.T @ modes, np.diag(np.diag(modes.T @ modes)))
    assert_min_spacing_within_modes(positions, modes, min_distance=1.1)

def test_zonal_fast_generation_clique():
    positions = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, np.sqrt(3.0) / 2.0],
    ])

    gen = ZonalFastBasisGenerator(positions, min_distance=1.01)
    modes = gen.generate()

    assert modes.shape == (3, 3)
    assert np.allclose(modes, np.eye(3))

def test_zonal_fast_single_mode_when_distance_is_small():
    positions = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
    ])

    gen = ZonalFastBasisGenerator(positions, min_distance=0.5)
    modes = gen.generate()

    assert modes.shape == (4, 1)
    assert np.allclose(modes[:, 0], np.ones(4))

def test_zonal_fast_mode_subset_and_errors():
    positions = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
    ])

    gen = ZonalFastBasisGenerator(positions, min_distance=1.1)
    subset = gen.generate(n_modes=1)
    assert subset.shape == (3, 1)

    with pytest.raises(ValueError, match="full basis only contains"):
        gen.generate(n_modes=3)

    with pytest.raises(ValueError, match="non-negative"):
        ZonalFastBasisGenerator(positions, min_distance=-0.1)


def test_zonal_fast_uses_modulo_coloring_on_grid_positions():
    axis = np.arange(5, dtype=float)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    positions = np.column_stack((xx.ravel(), yy.ravel()))

    gen = ZonalFastBasisGenerator(positions, min_distance=2.0)
    modes = gen.generate()

    assert modes.shape == (25, 4)
    assert_min_spacing_within_modes(positions, modes, min_distance=2.0)

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
