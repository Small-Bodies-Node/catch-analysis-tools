"""Test astrometric calibration."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ...services.calibration.astrometry import AstrometricCalibrationError, run_pipeline
from ...services.calibration.astrometry.clean import cleanup_files

RA_DEG = 263.0
DEC_DEG = 34.5


@pytest.mark.integration
@pytest.mark.skipif(
    shutil.which("solve-field") is None or "ASTROMETRY_CONFIG" not in os.environ,
    reason="solve-field or astrometry config not available",
)
@pytest.mark.parametrize("use_ra_dec", ([True, False]))
def test_run_pipeline(use_ra_dec):
    # Use a real sky FITS image committed to the repo
    input_fits = Path(__file__).parent.parent / "data" / "Comet_65P_Gunn_LONEOS.fits"
    assert input_fits.exists(), "test image is missing"

    # Run solve-field
    results = run_pipeline(
        str(input_fits), 51.0, 17.0, use_ra_dec, 2.4, search_radius=2
    )

    # input file will be copied
    assert results["wcs_image_url"] != input_fits
    assert np.isclose(results["center_ra_deg"], 51.1350620)
    assert np.isclose(results["center_dec_deg"], 17.3879501)
    assert np.isclose(results["pixel_scale"], 2.5293)


@pytest.mark.skipif(
    shutil.which("solve-field") is None or "ASTROMETRY_CONFIG" not in os.environ,
    reason="solve-field or astrometry config not available",
)
def test_run_pipeline_cannot_solve():
    # test solve_field, but no solution
    input_file = Path(__file__).parent.parent / "data" / "Comet_65P_Gunn_LONEOS.fits"

    with pytest.raises(
        AstrometricCalibrationError, match="solve-field did not produce a WCS solution"
    ):
        run_pipeline(
            input_file,
            52,
            16,
            True,
            2.53,
            scale_low=2.52,
            scale_high=2.54,
            search_radius=0.1,
        )


@pytest.mark.skipif(
    shutil.which("solve-field") is None, reason="solve-field not available"
)
def test_run_pipeline_solve_field_non_zero_exit():
    with patch.dict(os.environ, {"ASTROMETRY_CONFIG": "/fake/astrometry/config"}):
        with pytest.raises(AstrometricCalibrationError, match="solve-field failed"):
            run_pipeline("input.fits", RA_DEG, DEC_DEG, 1, 2)


@pytest.mark.skipif(
    shutil.which("solve-field") is None, reason="solve-field not available"
)
def test_run_pipeline_no_config_in_environ(monkeypatch):
    monkeypatch.delenv("ASTROMETRY_CONFIG", raising=False)
    with pytest.raises(
        AstrometricCalibrationError, match="ASTROMETRY_CONFIG is not set"
    ):
        run_pipeline("input.fits", RA_DEG, DEC_DEG, 1, 2)


def test_cleanup_files():
    with tempfile.TemporaryDirectory() as tmp_path:
        file_base = Path(tmp_path) / "testfile"
        extensions = [
            ".axy",
            ".corr",
            ".match",
            ".new",
            ".rdls",
            ".solved",
            "-ngc.png",
            "-objs.png",
            "-indx.png",
            "-indx.xyls",
        ]
        for ext in extensions:
            (file_base.with_name(file_base.name + ext)).write_text("tmp")
        cleanup_files(str(file_base))
        for ext in extensions:
            assert not (file_base.with_name(file_base.name + ext)).exists()
