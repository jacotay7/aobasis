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

    # ... existing code ...

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

def test_hadamard_generation(grid):
    gen = HadamardBasisGenerator(grid)
    modes = gen.generate(n_modes=8)
    assert modes.shape == (grid.shape[0], 8)
    # Entries should be 1 or -1
    assert np.all(np.isin(modes, [1, -1]))

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
