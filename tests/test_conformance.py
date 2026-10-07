"""aobasis follows the AO stack conventions (aocore CONVENTIONS.md, section 5).

aobasis owns the Zernike basis for the whole stack, so its orientation and
normalization are checked against the shared contract.
"""

import numpy as np
import pytest

from aobasis import ZernikeBasisGenerator

conformance = pytest.importorskip("aocore.conformance")


def test_zernike_basis_is_noll_normalised_with_tip_along_x():
    def zernike(j, y, x):
        positions = np.column_stack((x, y))
        return ZernikeBasisGenerator(positions, pupil_radius=1.0).generate(j)[:, j - 1]

    conformance.check_zernike_basis(zernike)
