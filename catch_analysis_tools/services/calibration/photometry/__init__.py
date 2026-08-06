import base64
import io
import uuid
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sep
from astropy.io import fits

from ....exceptions import PhotometricCalibrationError
from ..load_wcs import load_wcs
from .attach_sky_coordinates import attach_sky_coordinates
from .calibrate_photometric_zero_point import calibrate_photometric_zero_point
from .detect import detect_sources_on_background_subtracted_image
from .plot import plot_color_correction, plot_photometric_matches
from .write import write_photometric_calibration_output

matplotlib.use("Agg")


def _get_matched_indices(objids, source_count: int):
    """
    Return indices of sources matched to catalog objects.
    """
    if hasattr(objids, "mask"):
        return np.where(~objids.mask)[0]

    return np.arange(source_count)


def _get_color_corrected_indices(color_mags, source_count: int):
    """
    Return indices of sources used for color correction.
    """
    if hasattr(color_mags, "mask"):
        return np.where(~color_mags.mask)[0]

    return np.arange(source_count)


def _encode_figure(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def _resolve_photometry_output_fits(image_file: str) -> str:
    """
    Build a persistent output path for the astrometrically solved FITS file.
    """

    out_dir = Path(__file__).resolve().parents[2] / "outputs" / "astrometry"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(image_file).stem
    unique_suffix = uuid.uuid4().hex[:8]
    return str(out_dir / f"{stem}_{unique_suffix}_photometry.fits")


def run_pipeline(
    input_fits: str,
    snr_threshold: float,
    aperture_radius: float,
    catalog: str,
    cal_band: str,
    color_index: str | None = None,
    return_plot: bool = False,
    plot_type: str = "color_correction",
) -> dict[str, Any]:
    """Photometrically calibrate an image.


    Parameters
    ----------
    input_fits : str
        The file to calibrate.

    snr_threshold : float
        Source detection threshold.

    aperture_radius : float
        Measure sources with this aperture radius.

    catalog : str
        Calibrate to this photometric catalog.  See
        `catch_analysis_tools.services.calibration.photometry.catalogs` for
        supported catalogs.

    cal_band : str
        Calibrate to this photometric band.  See
        `catch_analysis_tools.services.calibration.photometry.catalogs` for
        supported bands.

    color_index : str, optional
        Derive a color correction using this color index.  Must be a string of
        the form "band1-band2", e.g., "g-r".  See
        `catch_analysis_tools.services.calibration.photometry.catalogs` for
        supported bands.

    return_plot : bool, optional
        Set to true to return a plot.

    plot_type : str, optional
        Plot to return: "color_correction" or "image_overlay".


    Returns
    -------
    calibration : dict
        The calibration information and plot (if requested).

    """

    image = fits.getdata(input_fits).astype(np.float32)
    output_fits = _resolve_photometry_output_fits(input_fits)

    # generate and remove background
    bkg = sep.Background(image)
    image_sub = image - bkg.back()

    source_list = detect_sources_on_background_subtracted_image(
        image_sub=image_sub,
        bkg_err=float(bkg.globalrms),
        snr=snr_threshold,
        aperture_radius=aperture_radius,
    )

    wcs_solution = load_wcs(input_fits)
    source_list, sky_coords = attach_sky_coordinates(source_list, wcs_solution)

    try:
        calibration = calibrate_photometric_zero_point(
            sky_coords,
            source_list,
            catalog,
            cal_band,
            color_index=color_index,
        )
    except Exception as exc:
        raise PhotometricCalibrationError(
            f"Photometric calibration failed: {exc}"
        ) from exc

    zp = calibration["zp"]
    color_term = calibration["color_term"]
    unc = calibration["unc"]
    m = calibration["m"]
    m_inst = calibration["m_inst"]
    color_mags = calibration["color_mags"]
    objids = calibration["objids"]

    matched_idx = _get_matched_indices(objids, len(source_list))
    colored_idx = _get_color_corrected_indices(color_mags, len(source_list))

    plots = {}

    if return_plot:
        make_color = plot_type in {"color_correction", "all"}
        make_overlay = plot_type in {"image_overlay", "all"}

        if make_color:
            fig, _ = plot_color_correction(
                color_mags,
                m,
                m_inst,
                zp,
                color_term,
                color_index,
            )
            plots["color_correction"] = _encode_figure(fig)
            plt.close(fig)

        if make_overlay:
            fig, _ = plot_photometric_matches(
                image_sub,
                source_list,
                matched_idx,
                colored_idx,
            )
            plots["image_overlay"] = _encode_figure(fig)
            plt.close(fig)

    write_photometric_calibration_output(
        output_fits,
        image_sub,
        wcs_solution,
        source_list,
        matched_idx,
        colored_idx,
        zp,
        unc,
        catalog,
        cal_band,
        color_index,
        color_term=color_term,
    )

    result = {
        "photometry": {
            "zero_point": float(zp),
            "color_term": float(color_term),
            "uncertainty": float(unc),
        },
        "sources": {
            "detected": int(len(source_list)),
            "matched": int(len(matched_idx)),
        },
        "output_fits": output_fits,
    }

    if plots:
        result["plots"] = plots

    return result
