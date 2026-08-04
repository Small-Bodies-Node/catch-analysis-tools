import os
import tempfile
from collections import namedtuple
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from ...exceptions import PhotometricCalibrationError
from ...services.calibration.photometry import run_pipeline
from ...services.calibration.photometry.attach_sky_coordinates import (
    attach_sky_coordinates,
)
from ...services.calibration.photometry.calibrate_photometric_zero_point import (
    calibrate_photometric_zero_point,
)
from ...services.calibration.photometry.detect import (
    detect_sources_on_background_subtracted_image,
)
from ...services.calibration.photometry.write import (
    write_photometric_calibration_output,
)
from .load_wcs_test import synthetic_wcs  # noqa: F401


@pytest.fixture
def synthetic_image():
    np.random.seed(23)
    image = np.random.normal(loc=0, scale=1.0, size=(100, 100)).astype(np.float32)
    for i in range(-5, 6):
        for j in range(-5, 6):
            image[50 + i, 50 + j] += 20 * np.exp(-(i**2 + j**2) / (2 * 1.5**2))
    return image


def observe_stars(N):
    """Return a list of a few stars for calibration testing."""
    np.random.seed(2026)

    # calibration with color correction
    zp = 27.4
    C = 0.5

    def cal(gmr):
        return zp + C * gmr

    # generate stars' instrumental counts
    np.random.seed(2026)
    gmr = 1.5 * np.random.rand(N)
    m = -2.5 * np.log10(40 * np.random.rand(N))
    m_inst = m - cal(gmr)
    counts = 10 ** (-0.4 * m_inst)

    Stars = namedtuple(
        "Stars", ["N", "zp", "color_term", "gmr", "m", "m_inst", "counts"]
    )
    return Stars(N=N, zp=zp, color_term=C, gmr=gmr, m=m, m_inst=m_inst, counts=counts)


@contextmanager
def mock_calviacat(N):
    """Mock PanSTARRS1 catalog methods for calibration testing."""
    stars = observe_stars(N)
    # calibration without color correction
    dm = stars.m - stars.m_inst
    zp_mean = np.mean(dm)
    zp_median = np.median(dm)

    # create mocked PanSTARRS1 catalog results
    mock_cat = MagicMock()
    mock_cat.search.return_value = (np.arange(stars.N),)
    mock_cat.xmatch.return_value = (np.arange(stars.N), 0.1 * np.ones(stars.N))
    mock_cat.cal_color.return_value = (
        stars.zp,
        stars.color_term,
        0.01,
        stars.m,
        stars.gmr,
        None,
    )
    mock_cat.cal_constant.return_value = (zp_mean, zp_median, 0.01, stars.m, None)

    with patch(
        "catch_analysis_tools.services.calibration.photometry"
        ".calibrate_photometric_zero_point.cvc"
    ) as mock_cvc:
        mock_cvc.PanSTARRS1.return_value = mock_cat
        yield mock_cvc


@pytest.fixture
def temporary_file():
    """Provides a temporary file name, and removes it after the test."""

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        fn = tmp.name

    yield fn

    if os.path.exists(fn):
        os.remove(fn)


def test_detect_sources(synthetic_image):
    bkg_err, snr = 1.0, 5.0
    source_list = detect_sources_on_background_subtracted_image(
        synthetic_image, bkg_err, snr, 8
    )
    assert isinstance(source_list, pd.DataFrame)
    assert len(source_list) == 1
    assert np.isclose(source_list["x"], 50, rtol=0.01)
    assert np.isclose(source_list["y"], 50, rtol=0.01)

    area = np.pi * 8**2
    expected_error = np.sqrt(area * 1.0**2)
    assert np.isclose(source_list["aperture_sum"], 282.637, atol=expected_error)


def test_detect_sources_no_sources(synthetic_image):
    with pytest.raises(PhotometricCalibrationError, match="No sources were detected"):
        detect_sources_on_background_subtracted_image(0 * synthetic_image, 1.0, 5, 8)


def test_detect_sources_no_postivie_sources(synthetic_image):
    im = synthetic_image - 10
    with pytest.raises(
        PhotometricCalibrationError,
        match="No detected sources have positive aperture_sum",
    ):
        detect_sources_on_background_subtracted_image(im, 0.1, 1, 8)


def test_pixel_to_sky(synthetic_wcs):  # noqa: F811
    df = pd.DataFrame({"aperture_sum": [1000, 2000, 500]})
    coords = SkyCoord([150.0, 150.01, 150.02], [2.0, 2.01, 2.02], unit="deg")
    x, y = synthetic_wcs.world_to_pixel(coords)
    df["x"] = x
    df["y"] = y
    source_list, sky_coords = attach_sky_coordinates(df, synthetic_wcs)

    assert "RA" in source_list
    assert "Dec" in source_list
    assert np.allclose(source_list["RA"], coords.ra.deg)
    assert np.allclose(source_list["Dec"], coords.dec.deg)
    assert u.allclose(sky_coords.ra, coords.ra)
    assert u.allclose(sky_coords.dec, coords.dec)


def test_calibrate_photometry_with_color_index():
    stars = observe_stars(10)  # test minimum number of stars
    df = pd.DataFrame({"aperture_sum": stars.counts})
    coords = SkyCoord(
        150 + 0.01 * np.arange(stars.N), 2 + 0.01 * np.arange(stars.N), unit="deg"
    )

    with mock_calviacat(stars.N):
        result = calibrate_photometric_zero_point(coords, df)

    assert result["catalog"] == "PanSTARRS1"
    assert result["cal_band"] == "r"
    assert result["color_index"] == "g-r"
    assert result["zp"] == stars.zp
    assert result["color_term"] == stars.color_term
    assert result["unc"] == 0.01
    assert np.allclose(result["m_inst"], stars.m_inst)
    assert np.allclose(result["m"], stars.m)
    assert np.allclose(result["color_mags"], stars.gmr)
    assert all(result["objids"] == np.arange(stars.N))
    assert np.allclose(result["distances"], 0.1 * np.ones(stars.N))
    assert len(result["m"]) == stars.N


def test_calibrate_photometry_without_color_index():
    stars = observe_stars(100)
    df = pd.DataFrame({"aperture_sum": stars.counts})
    coords = SkyCoord(
        150 + 0.01 * np.arange(stars.N), 2 + 0.01 * np.arange(stars.N), unit="deg"
    )

    with mock_calviacat(stars.N):
        result = calibrate_photometric_zero_point(coords, df, color_index=None)

    assert result["catalog"] == "PanSTARRS1"
    assert result["cal_band"] == "r"
    assert result["color_index"] is None
    # we expect the mean difference to be returned:
    assert np.isclose(result["zp"], np.mean(stars.m - stars.m_inst))
    assert result["color_term"] == 0
    assert result["unc"] == 0.01
    assert np.allclose(result["m_inst"], stars.m_inst)
    assert np.allclose(result["m"], stars.m)
    assert result["color_mags"] is None
    assert all(result["objids"] == np.arange(stars.N))
    assert np.allclose(result["distances"], 0.1 * np.ones(stars.N))

    assert len(result["m"]) == stars.N


def test_calibrate_photometry_not_enough_sources():
    # needs at least 10 sources
    df = pd.DataFrame({"aperture_sum": [1000, 2000, 500]})
    coords = SkyCoord([150.0, 150.01, 150.02], [2.0, 2.01, 2.02], unit="deg")

    # create mocked calviacat catalog results
    mock_cat = MagicMock()
    mock_cat.search.return_value = (np.arange(3),)
    mock_cat.xmatch.return_value = (np.array([1, 2, 3]), np.array([0.1, 0.2, 0.3]))
    mock_cat.cal_color.return_value = (
        25.0,
        0.05,
        0.01,
        np.array([20.1, 20.2, 20.3]),
        np.array([0.3, 0.2, 0.1]),
        None,
    )

    with patch(
        "catch_analysis_tools.services.calibration.photometry"
        ".calibrate_photometric_zero_point.cvc"
    ) as mock_cvc:
        mock_cvc.PanSTARRS1.return_value = mock_cat
        with pytest.raises(
            PhotometricCalibrationError, match="only 3 valid matched sources remain"
        ):
            calibrate_photometric_zero_point(coords, df)


def test_calibrate_photometry_no_xmatch():
    df = pd.DataFrame({"aperture_sum": [1000, 2000, 500]})
    coords = SkyCoord([150.0, 150.01, 150.02], [2.0, 2.01, 2.02], unit="deg")

    # create mocked calviacat catalog results
    mock_cat = MagicMock()
    mock_cat.search.return_value = (np.arange(3),)
    mock_cat.xmatch.return_value = None

    with patch(
        "catch_analysis_tools.services.calibration.photometry"
        ".calibrate_photometric_zero_point.cvc"
    ) as mock_cvc:
        mock_cvc.PanSTARRS1.return_value = mock_cat
        with pytest.raises(
            PhotometricCalibrationError,
            match="Photometric calibration failed: fewer than 10 catalog matches.",
        ):
            calibrate_photometric_zero_point(coords, df)


def test_calibrate_photometry_bad_source_list():
    stars = observe_stars(100)
    df = pd.DataFrame({"aperture_summmmm": stars.counts})
    coords = SkyCoord(
        150 + 0.01 * np.arange(stars.N), 2 + 0.01 * np.arange(stars.N), unit="deg"
    )

    with mock_calviacat(stars.N):
        with pytest.raises(ValueError):
            calibrate_photometric_zero_point(coords, df, color_index=None)

    df = pd.DataFrame({"aperture_sum": -stars.counts})
    with mock_calviacat(stars.N):
        with pytest.raises(ValueError):
            calibrate_photometric_zero_point(coords, df, color_index=None)


def test_calibrate_photometry_bad_catalog():
    stars = observe_stars(100)
    df = pd.DataFrame({"aperture_sum": stars.counts})
    coords = SkyCoord(
        150 + 0.01 * np.arange(stars.N), 2 + 0.01 * np.arange(stars.N), unit="deg"
    )

    with mock_calviacat(stars.N):
        with pytest.raises(ValueError, match="Invalid catalog: amazing"):
            calibrate_photometric_zero_point(
                coords, df, catalog="amazing", color_index=None
            )


def test_write_photometric_cal(synthetic_image, temporary_file):
    output_fits = temporary_file
    image = np.arange(100).reshape((10, 10))
    wcs_solution = WCS()

    # image with three stars
    new_image = np.zeros_like(synthetic_image)
    new_image += synthetic_image
    new_image += np.roll(synthetic_image, 20) / 2
    new_image += np.roll(synthetic_image, -20) * 2

    source_list = detect_sources_on_background_subtracted_image(new_image, 1.0, 5.0, 8)

    matched_idx = np.arange(len(source_list) - 1)
    color_corrected_idx = np.arange(len(source_list) - 2)
    zero_point = 23.4
    zero_point_uncertainty = 0.123
    catalog = "PanSTARRS1"
    cal_band = "r"
    color_index = "g-r"
    color_term = 0.432

    write_photometric_calibration_output(
        output_fits,
        image,
        wcs_solution,
        source_list,
        matched_idx,
        color_corrected_idx,
        zero_point,
        zero_point_uncertainty,
        catalog,
        cal_band,
        color_index,
        color_term,
    )

    with fits.open(output_fits) as hdul:
        assert hdul[0].header["CATALOG"] == catalog
        assert hdul[0].header["CAL_BAND"] == cal_band
        assert hdul[0].header["COLRINDX"] == str(color_index)
        assert np.isclose(hdul[0].header["ZP"], zero_point)
        assert np.isclose(hdul[0].header["COLRTERM"], color_term)
        assert np.isclose(hdul[0].header["ZP_UNC"], zero_point_uncertainty)

        assert np.allclose(hdul[0].data, image)

        # first table is source list
        tab = Table(hdul[1].data)
        assert len(tab) == 3
        assert np.allclose(tab["aperture_sum"], source_list["aperture_sum"])

        # second table are the matched sources
        tab = Table(hdul[2].data)
        assert len(tab) == 2
        assert np.allclose(tab["aperture_sum"], source_list["aperture_sum"][:2])

        # third table are the match sources with colors
        tab = Table(hdul[3].data)
        assert len(tab) == 1
        assert np.allclose(tab["aperture_sum"], source_list["aperture_sum"][:1])

    # test empty source_list
    write_photometric_calibration_output(
        output_fits,
        image,
        wcs_solution,
        source_list[:0],
        matched_idx,
        color_corrected_idx,
        zero_point,
        zero_point_uncertainty,
        catalog,
        cal_band,
        color_index,
        color_term,
    )

    with fits.open(output_fits) as hdul:
        assert len(hdul[1].data) == 0
        assert len(hdul[2].data) == 0
        assert len(hdul[3].data) == 0


def test_run_pipeline(synthetic_image, synthetic_wcs, temporary_file):  # noqa F811
    # image of 10 sources
    im = np.zeros_like(synthetic_image)
    for i in range(-1, 2):
        for j in range(-1, 2):
            im += np.roll(np.roll(synthetic_image, i * 15, 0), j * 15, 1)
    im += np.roll(synthetic_image, 30)

    fits.writeto(temporary_file, im, synthetic_wcs.to_header())
    stars = observe_stars(10)
    output_fits = ""

    try:
        with mock_calviacat(stars.N):
            result = run_pipeline(
                temporary_file,
                3,
                5,
                "PanSTARRS1",
                "r",
                color_index="g-r",
                return_plot=False,
            )
            output_fits = result["output_fits"]

        assert result["photometry"]["zero_point"] == stars.zp
        assert result["photometry"]["color_term"] == stars.color_term
        assert result["photometry"]["uncertainty"] == 0.01
        assert result["sources"]["detected"] == 10
        assert result["sources"]["matched"] == 10
        assert os.path.exists(result["output_fits"])
    finally:
        if os.path.exists(output_fits):
            os.remove(output_fits)


def test_run_pipeline_fail(synthetic_image, synthetic_wcs, temporary_file):  # noqa F811
    fits.writeto(temporary_file, synthetic_image, synthetic_wcs.to_header())

    with pytest.raises(PhotometricCalibrationError):
        with mock_calviacat(5):
            run_pipeline(
                temporary_file,
                3,
                5,
                "PanSTARRS1",
                "r",
                color_index="g-r",
                return_plot=False,
            )
