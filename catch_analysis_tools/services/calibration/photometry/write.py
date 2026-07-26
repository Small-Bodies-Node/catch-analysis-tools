import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS


def write_photometric_calibration_output(
    output_fits: str,
    image: np.ndarray,
    wcs_solution: WCS,
    source_list: pd.DataFrame,
    matched_idx: np.ndarray,
    color_corrected_idx: np.ndarray,
    zero_point: float,
    zero_point_uncertainty: float,
    catalog: str,
    cal_band: str,
    color_index: str,
    color_term: float,
):
    """
    Write a FITS file with photometric calibration metadata and source tables.
    """
    image_arr = np.asarray(image)

    primary_hdu = fits.PrimaryHDU(
        data=image_arr,
        header=wcs_solution.to_header(),
    )

    primary_hdu.header["CATALOG"] = catalog, "photometric catalog"
    primary_hdu.header["CAL_BAND"] = cal_band, "calibration band"
    primary_hdu.header["COLRINDX"] = str(color_index), "color index"
    primary_hdu.header["ZP"] = zero_point, "magnitude zero point"
    primary_hdu.header["COLORTRM"] = color_term, "zerp point color term"
    primary_hdu.header["ZP_UNC"] = zero_point_uncertainty, "stdev of residuals"

    # replace masked values with NaN
    source_list_clean = source_list.map(
        lambda x: x.filled(np.nan) if hasattr(x, "filled") else x
    )

    detected_hdu = fits.BinTableHDU(
        Table.from_pandas(source_list_clean),
        name="DETECTED_SOURCES",
    )

    if not source_list_clean.empty:
        matched_hdu = fits.BinTableHDU(
            Table.from_pandas(
                source_list_clean.iloc[matched_idx].reset_index(drop=True)
            ),
            name="SELECTED_STARS",
        )

        color_hdu = fits.BinTableHDU(
            Table.from_pandas(
                source_list_clean.iloc[color_corrected_idx].reset_index(drop=True)
            ),
            name="COLOR_CORRECTION_STARS",
        )
    else:
        matched_hdu = fits.BinTableHDU(name="SELECTED_STARS")
        color_hdu = fits.BinTableHDU(name="COLOR_CORRECTION_STARS")

    hdul = fits.HDUList(
        [
            primary_hdu,
            detected_hdu,
            matched_hdu,
            color_hdu,
        ]
    )

    hdul.writeto(output_fits, overwrite=True)
