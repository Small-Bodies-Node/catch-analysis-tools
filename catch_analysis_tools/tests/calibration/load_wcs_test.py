import os
import tempfile

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from ...services.calibration.load_wcs import load_wcs


@pytest.fixture
def synthetic_wcs():
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = "RA---TAN", "DEC--TAN"
    wcs.wcs.cunit = "deg", "deg"
    wcs.wcs.crpix = 50.0, 50.0
    wcs.wcs.crval = 150.0, 2.0
    wcs.wcs.pc = [[-0.000277778, 0], [0, 0.000277778]]
    wcs.array_shape = 100, 100
    return wcs


def test_load_wcs(synthetic_wcs):
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
        filename = tmp.name

    try:
        fits.PrimaryHDU(np.zeros((2, 2)), header=synthetic_wcs.to_header()).writeto(
            filename, overwrite=True
        )
        wcs = load_wcs(filename)
        assert isinstance(wcs, WCS)
        assert wcs.wcs.ctype[0] == "RA---TAN"
    finally:
        os.remove(filename)
