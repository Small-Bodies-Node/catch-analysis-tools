import os
import warnings

from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning


def load_wcs(output_wcs: str) -> WCS:
    """
    Load a WCS solution from a FITS file header.


    Parameters
    ----------
    output_wcs : str
        Path to the FITS file containing the WCS header from astrometry.net().


    Returns
    -------
    wcs_solution : astropy.wcs.WCS
        World coordinate system solution object.

    """

    if not os.path.exists(output_wcs):
        raise FileNotFoundError(f"WCS file not found: {output_wcs}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        with fits.open(output_wcs) as hdul:
            wcs_solution = WCS(hdul[0].header)

    return wcs_solution
