import numpy as np
import pandas as pd
import sep

from ....exceptions import PhotometricCalibrationError


def detect_sources_on_background_subtracted_image(
    image_sub: np.ndarray,
    bkg_err: float,
    snr: float,
    aperture_radius: float,
) -> pd.DataFrame:
    """Detect sources on image.


    Parameters
    ----------
    image_sub : array
        The background subtracted image.

    bkg_err : float
        The per pixel 1 sigma uncertainty on the background.

    snr : flaot
        Detection threshold as signal-to-noise ratio.

    aperture_radius : float
        Measure brightness with a circular aperture of this radius.


    Returns
    -------
    source_list : pandas.DataFrame

    """

    sep.set_sub_object_limit(500)

    sources = sep.extract(
        image_sub,
        thresh=snr,
        err=bkg_err,
        deblend_nthresh=16,
    )

    source_list = pd.DataFrame(sources)

    if source_list.empty:
        raise PhotometricCalibrationError(
            "No sources were detected in the WCS-solved image."
        )

    flux, flux_err, _ = sep.sum_circle(
        image_sub,
        source_list["x"],
        source_list["y"],
        aperture_radius,
        err=bkg_err,
    )

    source_list["aperture_sum"] = flux
    source_list["aperture_err"] = flux_err

    source_list = source_list[source_list["aperture_sum"] > 0].reset_index(drop=True)

    if source_list.empty:
        raise PhotometricCalibrationError(
            "No detected sources have positive aperture_sum."
        )

    return source_list
