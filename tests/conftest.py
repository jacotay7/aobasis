import pytest


def _cuda_device_count():
    try:
        import cupy
    except ImportError:
        return 0
    try:
        return cupy.cuda.runtime.getDeviceCount()
    except Exception:  # CuPy installed but no usable driver or device
        return 0


@pytest.fixture(scope="session")
def gpu():
    """Skip unless CuPy and a CUDA device are available; then the GPU test must pass."""
    if _cuda_device_count() == 0:
        pytest.skip("needs CuPy and a CUDA device")
