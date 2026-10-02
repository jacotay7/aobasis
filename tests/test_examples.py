"""The scripts in examples/ run (with small sizes)."""

import importlib.util
import runpy
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quickstart(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLES / "quickstart.py"), run_name="__main__")
    assert (tmp_path / "kl_m2c.npz").exists()
    assert "276 actuators x 50 modes" in capsys.readouterr().out


@pytest.mark.parametrize("basis", ["kl", "zernike", "fourier", "hadamard", "zonal", "zonal_fast"])
def test_make_m2c(basis, tmp_path):
    out = tmp_path / f"{basis}.npz"
    m2c = _load("make_m2c").main([basis, "--grid-size", "10", "--n-modes", "8", "-o", str(out)])
    assert m2c.shape[1] == 8 and out.exists()


def test_make_m2c_geometry_options(tmp_path):
    positions = tmp_path / "dm.txt"
    np.savetxt(positions, np.random.default_rng(0).uniform(-4, 4, (60, 2)))
    m2c = _load("make_m2c").main(["kl", "--positions", str(positions), "--centre", "--remove", "tiptilt"])
    assert m2c.shape == (60, 57)
    hexagonal = _load("make_m2c").main(["hadamard", "--hex-pitch", "1.0", "--obscuration", "0.2", "--selection", "balanced"])
    assert hexagonal.shape[0] == hexagonal.shape[1] + 1
    with pytest.raises(SystemExit):
        _load("make_m2c").main(["zonal_fast", "--grid-size", "10", "--n-modes", "500"])


def test_compare_bases():
    curves = _load("compare_bases").main(["--grid-size", "10", "--n-modes", "20"])
    assert curves["KL"][-1] <= curves["Zernike"][-1] <= 1


def test_phase_screens(tmp_path):
    screens = _load("phase_screens").main(["--grid-size", "10", "--n-screens", "5", "-o", str(tmp_path / "s.npy")])
    assert screens.shape[1] == 5 and np.allclose(screens.mean(axis=0), 0, atol=1e-9)


def test_dm_kl(tmp_path):
    m2c = _load("dm_kl").main(["--grid-size", "8", "--pixels", "24", "--n-modes", "10", "-o", str(tmp_path / "d.npz")])
    assert m2c.shape[1] == 10


def test_calibration():
    errors = _load("calibration").main(["--actuators-across", "9", "--subapertures", "8", "--repeats", "4"])
    assert errors["Hadamard"] < errors["zonal"]


def test_benchmark():
    _load("benchmark").main(["--grid-sizes", "8", "--n-modes", "10"])
