"""save/load keep parameters, options, eigenvalues and version; FITS round trip (#30)."""

import numpy as np
import pytest

import aobasis
from aobasis import (
    BasisGenerator,
    KLBasisGenerator,
    ZernikeBasisGenerator,
    ZonalFastBasisGenerator,
    make_circular_actuator_grid,
)


@pytest.fixture
def grid():
    return make_circular_actuator_grid(telescope_diameter=10.0, grid_size=12)


def test_npz_keeps_metadata(grid, tmp_path):
    gen = KLBasisGenerator(grid, fried_parameter=0.12, outer_scale=np.inf, wavelength=1.65e-6)
    gen.generate(15, ignore_piston=True, remove=["tiptilt", np.ones(grid.shape[0])], normalize="rms")
    gen.save(tmp_path / "kl.npz")
    loaded = BasisGenerator.load(tmp_path / "kl.npz")
    assert loaded.basis_type == "KLBasisGenerator"
    assert loaded.aobasis_version == aobasis.__version__
    assert loaded.parameters == {
        "fried_parameter": 0.12,
        "outer_scale": float("inf"),
        "use_gpu": False,
        "r0_wavelength": 500e-9,
        "wavelength": 1.65e-6,
    }
    assert loaded.generate_options == {
        "n_modes": 15,
        "ignore_piston": True,
        "orthonormalize": False,
        "remove": ["tiptilt", {"array_shape": [grid.shape[0]]}],
        "normalize": "rms",
    }
    assert np.allclose(loaded.eigenvalues, gen.eigenvalues)
    assert np.allclose(loaded.modes, gen.modes)


def test_resaving_a_loaded_basis_keeps_metadata(grid, tmp_path):
    gen = ZonalFastBasisGenerator(grid, min_distance=2.0)
    gen.generate()
    gen.save(tmp_path / "a.npz")
    BasisGenerator.load(tmp_path / "a.npz").save(tmp_path / "b.npz")
    again = BasisGenerator.load(tmp_path / "b.npz")
    assert again.basis_type == "ZonalFastBasisGenerator"
    assert again.parameters == {"min_distance": 2.0}
    assert again.generate_options == {"n_modes": None, "normalize": None}
    assert again.eigenvalues is None


def test_old_files_without_metadata_still_load(grid, tmp_path):
    modes = np.eye(grid.shape[0])[:, :3]
    np.savez(tmp_path / "old.npz", modes=modes, positions=grid, basis_type="ZonalBasisGenerator")
    loaded = BasisGenerator.load(tmp_path / "old.npz")
    assert loaded.parameters == {} and loaded.generate_options == {}
    assert loaded.aobasis_version is None and loaded.eigenvalues is None


def test_fits_round_trip(grid, tmp_path):
    pytest.importorskip("astropy")
    gen = KLBasisGenerator(grid)
    gen.generate(12, ignore_piston=True)
    gen.save_fits(tmp_path / "kl.fits")
    loaded = KLBasisGenerator.load_fits(tmp_path / "kl.fits")
    assert np.allclose(loaded.modes, gen.modes)
    assert np.allclose(loaded.positions, grid)
    assert np.allclose(loaded.eigenvalues, gen.eigenvalues)
    assert loaded.parameters["outer_scale"] == 30.0
    assert loaded.generate_options["ignore_piston"] is True

    from astropy.io import fits

    with fits.open(tmp_path / "kl.fits") as hdus:
        assert hdus[0].data.shape == (grid.shape[0], 12)
        assert hdus[0].header["BASIS"] == "KLBasisGenerator"
    with pytest.raises(ValueError, match="KLBasisGenerator"):
        ZernikeBasisGenerator.load_fits(tmp_path / "kl.fits")
    with pytest.raises(OSError):
        gen.save_fits(tmp_path / "kl.fits")  # no overwrite by default
    gen.save_fits(tmp_path / "kl.fits", overwrite=True)


def test_fits_without_astropy_names_the_extra(grid, tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "astropy", None)
    monkeypatch.setitem(sys.modules, "astropy.io", None)
    gen = ZernikeBasisGenerator(grid, pupil_radius=5.0)
    gen.generate(3)
    with pytest.raises(ImportError, match=r"aobasis\[fits\]"):
        gen.save_fits(tmp_path / "z.fits")
