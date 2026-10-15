from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from PIL import Image
from pyavm import AVM
from reproject import reproject_interp


def render_north_up_preview(filename: str) -> bytes:
    """Derive a display PNG from this artifact's full WCS; preserve science pixels.

    The display contract is ICRS TAN, native dimensions/scales, north up/east
    left, centered on the native array center. AVM describes the rendered PNG,
    rather than the native detector geometry or a translated old solution.
    """
    with fits.open(filename) as hdus:
        hdu = next(hdu for hdu in hdus if hdu.data is not None and hdu.data.ndim == 2)
        data = np.asarray(hdu.data, dtype=float)
        native = WCS(hdu.header).celestial
        if not native.has_celestial:
            raise ValueError("Cannot render an oriented preview without celestial WCS")
        height, width = data.shape
        center = native.pixel_to_world((width - 1) / 2, (height - 1) / 2).icrs
        scales = native.proj_plane_pixel_scales()
        target = WCS(naxis=2)
        target.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        target.wcs.radesys = "ICRS"
        target.wcs.crpix = [(width + 1) / 2, (height + 1) / 2]
        target.wcs.crval = [center.ra.deg, center.dec.deg]
        target.wcs.cdelt = [-abs(scales[0].value), abs(scales[1].value)]
        target.pixel_shape = (width, height)
        rendered = reproject_interp(
            (data, native),
            target,
            shape_out=data.shape,
            return_footprint=False,
            order="nearest-neighbor",
        )
    scaled = (
        ZScaleInterval()(rendered) * 255
        if np.isfinite(rendered).any()
        else np.zeros(data.shape)
    )
    pixels = np.nan_to_num(scaled).clip(0, 255).astype(np.uint8)[::-1]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "preview.png"
        Image.fromarray(pixels).save(path)
        AVM.from_wcs(target, shape=data.shape).embed(str(path), str(path))
        return path.read_bytes()
