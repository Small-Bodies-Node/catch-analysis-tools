import calviacat as cvc
import numpy as np
import pandas as pd

from ....exceptions import PhotometricCalibrationError
from .catalogs import catalogs


def calibrate_photometric_zero_point(
    sky_coords,
    source_list: pd.DataFrame,
    catalog: str = "PanSTARRS1",
    cal_band: str = "r",
    color_index: str | None = "g-r",
    catalog_db: str = ":memory:",
):
    """
    Calibrate instrumental magnitudes using a reference catalog.

    This performs photometric calibration only: catalog matching, instrumental
    magnitude calculation, zero-point estimation, and (optinal) color-term
    correction.


    Parameters
    ----------
    sky_coords : astropy.coordinates.SkyCoord
        Sky coordinates of detected sources.

    source_list : pandas.DataFrame
        Source table containing aperture fluxes in the ``aperture_sum`` column.

    catalog : str, optional
        Name of the reference catalog class in calviacat. Default is
        ``"PanSTARRS1"``.

    obs_band : str, optional
        Observed image band label. Used to construct the color index.

    cal_band : str, optional
        Reference catalog band used for calibration.

    catalog_db : str, optional
        Local catalog database path.


    Returns
    -------
    dict
        Photometric calibration results containing zero point, color term,
        calibrated magnitudes, matched object IDs, and match distances.

    """

    if "aperture_sum" not in source_list.columns:
        raise ValueError("source_list must contain an 'aperture_sum' column.")

    aperture_sum = source_list["aperture_sum"].to_numpy()

    if np.any(aperture_sum <= 0):
        raise ValueError(
            "All aperture_sum values must be positive before magnitude calibration."
        )

    if catalog not in catalogs:
        raise ValueError(f"Invalid catalog: {catalog}.")

    CatalogClass = getattr(cvc, catalog)
    cat = CatalogClass(catalog_db)

    results = cat.search(sky_coords)
    if len(results[0]) < 500:
        cat.fetch_field(sky_coords)

    xmatch_result = cat.xmatch(sky_coords)

    if xmatch_result is None:
        raise PhotometricCalibrationError(
            "Photometric calibration failed: fewer than 10 catalog matches."
        )

    objids, distances = xmatch_result

    aperture_sum = np.asarray(source_list["aperture_sum"].values, dtype=float)
    valid_flux = np.isfinite(aperture_sum) & (aperture_sum > 0)

    objids = objids[valid_flux]
    distances = distances[valid_flux]
    aperture_sum = aperture_sum[valid_flux]

    m_inst = -2.5 * np.log10(aperture_sum)

    valid_m_inst = np.isfinite(m_inst)

    objids = objids[valid_m_inst]
    distances = distances[valid_m_inst]
    m_inst = m_inst[valid_m_inst]

    if len(m_inst) < 10:
        raise PhotometricCalibrationError(
            f"Photometric calibration failed: only {len(m_inst)} valid matched sources "
            "remain after filtering."
        )

    if color_index is None:
        zp_mean, zp_median, zp_unc, m_cal, colors = cat.cal_constant(
            objids, m_inst, cal_band
        )
        zp = zp_mean
        color_term = 0
    else:
        zp, color_term, zp_unc, m_cal, colors, _ = cat.cal_color(
            objids,
            m_inst,
            cal_band,
            color_index,
        )

    return {
        "catalog": catalog,
        "cal_band": cal_band,
        "color_index": color_index,
        "zp": zp,
        "color_term": color_term,
        "unc": zp_unc,
        "m_inst": m_inst,
        "m": m_cal,
        "color_mags": None if color_index is None else colors,
        "objids": objids,
        "distances": distances,
    }
