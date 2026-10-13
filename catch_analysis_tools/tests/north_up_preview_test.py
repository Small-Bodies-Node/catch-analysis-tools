import hashlib

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image
from pyavm import AVM

from ..services.render_north_up_preview import render_north_up_preview


@pytest.mark.parametrize(
    "angle,parity", [(0, -1), (31, -1), (180, -1), (0, 1), (31, 1), (180, 1)]
)
def test_preview_is_north_up_and_preserves_scientific_array(tmp_path, angle, parity):
    native = WCS(naxis=2)
    native.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    native.wcs.crval = [359.99, -16.2]
    native.wcs.crpix = [20.5, 30.5]
    rotation = np.deg2rad(angle)
    native.wcs.cd = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
    ) @ np.diag([parity * 0.00014, 0.00014])
    yy, xx = np.mgrid[:33, :57]
    data = np.exp(-((xx - 28) ** 2 + (yy - 16) ** 2) / 4)
    path = tmp_path / "science.fits"
    fits.PrimaryHDU(data, native.to_header(relax=True)).writeto(path)
    original = hashlib.sha256(path.read_bytes()).digest()
    preview = tmp_path / "preview.png"
    preview.write_bytes(render_north_up_preview(str(path)))
    assert Image.open(preview).size == (57, 33)
    assert hashlib.sha256(path.read_bytes()).digest() == original
    target = AVM.from_image(str(preview)).to_wcs()
    center = target.pixel_to_world(28, 16)
    import astropy.units as u

    north = target.world_to_pixel(center.directional_offset_by(0 * u.deg, 5 * u.arcsec))
    east = target.world_to_pixel(center.directional_offset_by(90 * u.deg, 5 * u.arcsec))
    assert north[1] > 16 and abs(north[0] - 28) < 1e-5
    assert east[0] < 28 and abs(east[1] - 16) < 1e-5
    np.testing.assert_allclose(
        target.pixel_to_world(28, 16)
        .icrs.separation(native.pixel_to_world(28, 16).icrs)
        .arcsec,
        0,
        atol=1e-5,
    )


def test_preview_rejects_missing_celestial_wcs(tmp_path):
    path = tmp_path / "science.fits"
    fits.PrimaryHDU(np.ones((10, 20))).writeto(path)
    with pytest.raises(ValueError, match="celestial WCS"):
        render_north_up_preview(str(path))
