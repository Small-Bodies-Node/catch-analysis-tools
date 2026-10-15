import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ..handlers.calibration import astrometry as astrometry_handler
from ..handlers.calibration import photometry as photometry_handler
from ..services import generated_images, result_cache


def _download_response(content=b"FITS"):
    response = SimpleNamespace(content=content)
    response.raise_for_status = MagicMock()
    return response


def test_upload_generated_file_streams_to_configured_bucket(tmp_path, monkeypatch):
    artifact = tmp_path / "result.fits"
    artifact.write_bytes(b"FITS")
    s3_client = MagicMock()

    monkeypatch.setenv("CAT_IMAGES_BUCKET", "images-bucket")
    monkeypatch.setenv("CAT_IMAGES_BASE_URL", "https://images.example")
    monkeypatch.setenv("CAT_IMAGES_PREFIX", "generated-images")

    with patch("boto3.client", return_value=s3_client):
        uploaded = generated_images.upload_generated_file(
            str(artifact),
            "calibration-astrometry",
            "astrometry.fits",
            "application/fits",
        )

    assert uploaded["url"].startswith(
        "https://images.example/generated-images/calibration-astrometry/"
    )
    assert uploaded["key"].endswith("-astrometry.fits")
    s3_client.upload_file.assert_called_once()
    assert s3_client.upload_file.call_args.kwargs["ExtraArgs"]["ContentType"] == (
        "application/fits"
    )


def test_upload_generated_bytes_publishes_png(monkeypatch):
    s3_client = MagicMock()
    monkeypatch.setenv("CAT_IMAGES_BUCKET", "images-bucket")
    monkeypatch.setenv("CAT_IMAGES_BASE_URL", "https://images.example")

    with patch("boto3.client", return_value=s3_client):
        uploaded = generated_images.upload_generated_bytes(
            b"PNG",
            "calibration-photometry",
            "color-correction.png",
            "image/png",
        )

    assert uploaded["url"].endswith("-color-correction.png")
    assert s3_client.put_object.call_args.kwargs["Body"] == b"PNG"
    assert s3_client.put_object.call_args.kwargs["ContentType"] == "image/png"


def test_astrometry_handler_publishes_pipeline_fits(tmp_path):
    output_fits = tmp_path / "astrometry.fits"
    output_fits.write_bytes(b"SOLVED")
    config = astrometry_handler.validate(
        {"image_url": "https://example.org/input.fits"}
    )

    with (
        patch.object(
            astrometry_handler.requests,
            "get",
            return_value=_download_response(),
        ),
        patch.object(
            astrometry_handler,
            "run_pipeline",
            return_value={
                "output_fits": str(output_fits),
                "center_ra_deg": 51.1,
                "center_dec_deg": 17.4,
                "pixel_scale": 2.5,
            },
        ) as run_pipeline,
        patch.object(
            astrometry_handler,
            "upload_generated_file",
            return_value={
                "url": "https://images.example/astrometry.fits",
                "key": "generated/astrometry.fits",
            },
        ) as upload_file,
        patch.object(
            astrometry_handler,
            "publish_preview",
            return_value={
                "preview_url": "https://images.example/astrometry-preview.png",
                "preview_s3_key": "generated/astrometry-preview.png",
            },
        ) as publish_preview,
    ):
        result = astrometry_handler._run_astrometry_uncached(config)

    assert result["status"] == "success"
    assert result["wcs_image_url"] == "https://images.example/astrometry.fits"
    assert result["wcs_image_s3_key"] == "generated/astrometry.fits"
    assert result["preview_url"] == "https://images.example/astrometry-preview.png"
    publish_preview.assert_called_once_with(str(output_fits), "calibration-astrometry")
    assert not output_fits.exists()
    run_pipeline.assert_called_once()
    upload_file.assert_called_once_with(
        str(output_fits),
        "calibration-astrometry",
        "astrometry.fits",
        "application/fits",
    )


def test_photometry_handler_returns_both_plots_and_calibrated_fits(tmp_path):
    output_fits = tmp_path / "photometry.fits"
    output_fits.write_bytes(b"CALIBRATED")
    config = photometry_handler.validate(
        {
            "wcs_image_url": "https://images.example/astrometry.fits",
            "color_term": "g-r",
        }
    )
    pipeline_result = {
        "photometry": {
            "zero_point": 25.0,
            "color_term": 0.12,
            "uncertainty": 0.03,
        },
        "sources": {"detected": 8, "matched": 5},
        "output_fits": str(output_fits),
        "plots": {
            "color_correction": base64.b64encode(b"COLOR").decode(),
            "image_overlay": base64.b64encode(b"OVERLAY").decode(),
        },
    }

    with (
        patch.object(
            photometry_handler.requests,
            "get",
            return_value=_download_response(),
        ),
        patch.object(
            photometry_handler,
            "run_pipeline",
            return_value=pipeline_result,
        ) as run_pipeline,
        patch.object(
            photometry_handler,
            "upload_generated_bytes",
            side_effect=[
                {"url": "https://images.example/color.png", "key": "color.png"},
                {"url": "https://images.example/overlay.png", "key": "overlay.png"},
            ],
        ) as upload_bytes,
        patch.object(
            photometry_handler,
            "upload_generated_file",
            return_value={
                "url": "https://images.example/photometry.fits",
                "key": "photometry.fits",
            },
        ) as upload_file,
        patch.object(
            photometry_handler,
            "publish_preview",
            return_value={
                "preview_url": "https://images.example/photometry-preview.png",
                "preview_s3_key": "generated/photometry-preview.png",
            },
        ) as publish_preview,
    ):
        result = photometry_handler._run_photometry_uncached(config)

    assert result["status"] == "success"
    assert result["preview_url"] == "https://images.example/photometry-preview.png"
    publish_preview.assert_called_once_with(str(output_fits), "calibration-photometry")
    assert result["photometry"] == {
        "zero_point": 25.0,
        "color_term": 0.12,
        "uncertainty": 0.03,
        "color_index": "g-r",
        "catalog": "PanSTARRS1",
        "cal_band": "r",
    }
    assert result["plots"]["color_correction"]["url"].endswith("color.png")
    assert result["plots"]["image_overlay"]["url"].endswith("overlay.png")
    assert result["photometry_image_url"].endswith("photometry.fits")
    assert upload_bytes.call_count == 2
    assert upload_bytes.call_args_list[0].args[0] == b"COLOR"
    assert upload_bytes.call_args_list[1].args[0] == b"OVERLAY"
    upload_file.assert_called_once()
    assert run_pipeline.call_args.kwargs["return_plot"] is True
    assert run_pipeline.call_args.kwargs["plot_type"] == "all"
    assert run_pipeline.call_args.kwargs["color_index"] == "g-r"
    assert not output_fits.exists()


def test_photometry_validation_uses_one_input_contract():
    config = photometry_handler.validate(
        {
            "wcs_image_url": "https://images.example/astrometry.fits",
            "return_plot": True,
            "plot_type": "image_overlay",
        }
    )

    assert config["color_term"] == "g-r"
    assert "return_plot" not in config
    assert "plot_type" not in config


def test_photometry_upload_failure_is_json():
    with patch.object(
        photometry_handler,
        "get_or_compute",
        side_effect=generated_images.GeneratedArtifactUploadError("upload failed"),
    ):
        response = photometry_handler.handler(
            {"wcs_image_url": "https://images.example/astrometry.fits"}
        )

    payload = json.loads(response.get_data(as_text=True))
    assert response.status_code == 500
    assert payload["status"] == "artifact_upload_failed"
    assert payload["message"] == "upload failed"


def test_cache_is_disabled_when_no_bucket_is_configured(monkeypatch):
    for variable in ("CAT_CACHE_BUCKET", "TF_VAR_CAT_CACHE_BUCKET_NAME"):
        monkeypatch.delenv(variable, raising=False)
    compute = MagicMock(return_value={"status": "success"})

    result = result_cache.get_or_compute("example-route", {"value": 1}, compute)

    compute.assert_called_once_with()
    assert result["cache"] == {
        "enabled": False,
        "hit": False,
        "key": None,
    }


def test_cache_key_is_stable_for_equivalent_inputs():
    first = result_cache.make_cache_key(
        "calibration-photometry", {"catalog": "PanSTARRS1", "value": 1}
    )
    second = result_cache.make_cache_key(
        "calibration-photometry", {"value": 1, "catalog": "PanSTARRS1"}
    )

    assert first == second


def test_cache_hit_does_not_regenerate_artifacts(monkeypatch):
    monkeypatch.setenv("CAT_IMAGES_BASE_URL", "https://images.example")
    cached_payload = {
        "status": "success",
        "photometry_image_s3_key": "photometry/result.fits",
        "preview_s3_key": "photometry/preview.png",
        "plots": {
            "color_correction": {"s3_key": "photometry/color.png"},
            "image_overlay": {"s3_key": "photometry/overlay.png"},
        },
    }
    compute = MagicMock()

    with (
        patch.object(result_cache, "_cache_bucket", return_value="cache-bucket"),
        patch.object(result_cache, "_read_json", return_value=cached_payload),
    ):
        result = result_cache.get_or_compute(
            "calibration-photometry",
            {"wcs_image_url": "https://images.example/input.fits"},
            compute,
        )

    compute.assert_not_called()
    assert result["cache"]["hit"] is True
    assert result["photometry_image_url"].endswith("photometry/result.fits")


def test_cache_refreshes_calibration_artifact_urls(monkeypatch):
    monkeypatch.setenv("CAT_IMAGES_BASE_URL", "https://new-images.example")
    payload = {
        "wcs_image_s3_key": "astrometry/result.fits",
        "photometry_image_s3_key": "photometry/result.fits",
        "preview_s3_key": "photometry/preview.png",
        "plots": {
            "color_correction": {"s3_key": "photometry/color.png"},
            "image_overlay": {"s3_key": "photometry/overlay.png"},
        },
    }

    refreshed = result_cache._refresh_image_urls(payload)

    assert refreshed["wcs_image_url"].endswith("astrometry/result.fits")
    assert refreshed["photometry_image_url"].endswith("photometry/result.fits")
    assert refreshed["preview_url"].endswith("photometry/preview.png")
    assert refreshed["plots"]["color_correction"]["url"].endswith(
        "photometry/color.png"
    )
    assert refreshed["plots"]["image_overlay"]["url"].endswith("photometry/overlay.png")
