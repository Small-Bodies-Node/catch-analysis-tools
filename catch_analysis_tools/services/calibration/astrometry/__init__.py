import os
import uuid
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.io import fits

from ....exceptions import AstrometricCalibrationError
from ..load_wcs import load_wcs
from .clean import cleanup_files
from .solve_field import solve_field
from .write import write_astrometry_output


def _resolve_astrometry_output_fits(image_file: str) -> str:
    """
    Build a persistent output path for the astrometrically solved FITS file.
    """

    out_dir = Path(__file__).resolve().parents[2] / "outputs" / "astrometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_file).stem
    unique_suffix = uuid.uuid4().hex[:8]
    return str(out_dir / f"{stem}_{unique_suffix}_astrometry.fits")


def run_pipeline(
    input_file: str,
    ra: float | None,
    dec: float | None,
    use_ra_dec: bool,
    pixel_scale: float,
    scale_low: float | None = None,
    scale_high: float | None = None,
    search_radius: float = 2,
    output_file: str | None = None,
) -> dict[str, Any]:
    """Run the astrometric calibration service.


    Parameters
    ----------
    input_file : str
        The image file name to calibrate.

    ra: float or None
        Right ascension guess. (degrees)

    dec: float or None
        Declination guess. (degrees)

    use_ra_dec : bool
        ``True`` if ``ra`` and ``dec`` should be used as a guess.

    pixel_scale : float
        Pixel scale guess. (arcsec/pix)

    scale_low : float, optional
        Lower limit on pixel scale solution or ``None`` to use ``0.5 *
        pixel_scale``.  (arcsec/pix)

    scale_high : float, optional
        Upper limit on pixel scale solution or ``None`` to use ``2 *
        pixel_scale``.  (arcsec/pix)

    search_radius : float, optional
        Search radius around (ra, dec).  Ignored if ``use_ra_dec`` is ``False``.
        (degrees)

    output_file : string, optional
        Merge image data and WCS and save to this file.  Default is to generate
        a new file name.


    Returns
    -------
    result : dict

    """

    if output_file is None:
        output_file = _resolve_astrometry_output_fits(input_file)

    # astrometry.net generated files are based on the this prefix
    file_base = os.path.splitext(input_file)[0]

    # output_wcs is the expected astrometry.net result
    output_wcs = f"{file_base}.wcs"

    if not use_ra_dec:
        ra = None
        dec = None

    scale_low = pixel_scale * 0.5 if scale_low is None else scale_low
    scale_high = pixel_scale * 2.0 if scale_high is None else scale_high

    # solve_field may raise an exception
    try:
        finished = solve_field(
            input_file, ra, dec, scale_low, scale_high, search_radius
        )
    except RuntimeError as exc:
        raise AstrometricCalibrationError(str(exc)) from exc

    # in case astrometry.net runs but does not create an output file
    if finished:
        try:
            wcs_solution = load_wcs(output_wcs)
        except FileNotFoundError:
            raise AstrometricCalibrationError(
                "solve-field did not produce a WCS solution."
            )

    image = fits.getdata(input_file)
    write_astrometry_output(image, wcs_solution, output_file)

    cleanup_files(file_base)

    ny, nx = image.shape
    center_world = wcs_solution.pixel_to_world(nx / 2.0, ny / 2.0)

    # average pixel scale
    pixel_scale = np.mean(
        np.abs(u.Quantity(wcs_solution.proj_plane_pixel_scales(), "arcsec").value)
    )

    return {
        "wcs_image_url": output_file,
        "center_ra_deg": float(center_world.ra.deg),
        "center_dec_deg": float(center_world.dec.deg),
        "pixel_scale": pixel_scale,
    }
