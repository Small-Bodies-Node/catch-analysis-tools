import base64
import json
import logging
import os
from tempfile import NamedTemporaryFile
from uuid import uuid4

import requests
from flask import Response
from werkzeug.exceptions import BadRequest

from ...exceptions import InputValidationError
from ...services.calibration.photometry import PhotometricCalibrationError, run_pipeline
from ...services.calibration.photometry.catalogs import catalogs
from ...services.generated_images import (
    GeneratedArtifactUploadError,
    upload_generated_bytes,
    upload_generated_file,
)
from ...services.publish_preview import publish_preview
from ...services.result_cache import get_or_compute

logger = logging.getLogger(__name__)


def json_response(payload, status):
    return Response(json.dumps(payload), status=status, mimetype="application/json")


def validate(body):
    """Validate and normalize inputs before building the cache key."""
    wcs_image_url = body.get("wcs_image_url")
    if not wcs_image_url:
        raise InputValidationError("wcs_image_url is required")

    catalog = body.get("catalog", "PanSTARRS1")
    if catalog not in catalogs:
        raise InputValidationError(f"Unknown photometric catalog: {catalog}")

    config = {
        "wcs_image_url": wcs_image_url,
        "snr_threshold": body.get("snr_threshold", 3.0),
        "aperture_radius": body.get("aperture_radius", 7.0),
        "catalog": catalog,
        "cal_band": body.get("cal_band", "r"),
        "color_term": body.get("color_term", "g-r"),
    }
    cat = catalogs[catalog]

    if config["cal_band"] not in cat:
        raise InputValidationError(
            f"Cannot calibrate to {config['cal_band']} with catalog={catalog}"
        )

    color_index = config["color_term"]
    if color_index:
        bands = color_index.split("-")
        if len(bands) != 2:
            raise InputValidationError(
                "Invalid format for color_term: must be band1-band2."
            )

        if bands[0] == bands[1]:
            raise InputValidationError("Invalid color_term: both bands are the same.")

        for band in bands:
            if band not in cat:
                raise InputValidationError(
                    f"Invalid color_term: cannot calibrate to {band} "
                    f"with catalog={catalog}"
                )

    return config


def _run_photometry_uncached(config):
    try:
        response = requests.get(config["wcs_image_url"], timeout=60)
        response.raise_for_status()
    except requests.RequestException:
        raise BadRequest("Could not retrieve FITS image")

    output_fits = None
    with NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        result = run_pipeline(
            tmp_path,
            config["snr_threshold"],
            config["aperture_radius"],
            config["catalog"],
            config["cal_band"],
            color_index=config["color_term"],
            return_plot=True,
            plot_type="all",
        )
        output_fits = result["output_fits"]

        plot_files = {
            "color_correction": "color-correction.png",
            "image_overlay": "image-overlay.png",
        }
        published_plots = {}
        for plot_name, artifact_name in plot_files.items():
            encoded_plot = result.get("plots", {}).get(plot_name)
            if not encoded_plot:
                raise PhotometricCalibrationError(
                    f"Photometry did not produce the {plot_name} plot"
                )
            uploaded_plot = upload_generated_bytes(
                base64.b64decode(encoded_plot),
                "calibration-photometry",
                artifact_name,
                "image/png",
            )
            published_plots[plot_name] = {
                "url": uploaded_plot["url"],
                "s3_key": uploaded_plot["key"],
            }

        uploaded_fits = upload_generated_file(
            output_fits,
            "calibration-photometry",
            "photometry.fits",
            "application/fits",
        )

        photometry = dict(result["photometry"])
        photometry.update(
            {
                "color_index": config["color_term"],
                "catalog": config["catalog"],
                "cal_band": config["cal_band"],
            }
        )

        return {
            **publish_preview(output_fits, "calibration-photometry"),
            "status": "success",
            "photometry": photometry,
            "sources": result["sources"],
            "photometry_image_url": uploaded_fits["url"],
            "photometry_image_s3_key": uploaded_fits["key"],
            "plots": published_plots,
        }
    finally:
        if output_fits and os.path.exists(output_fits):
            os.remove(output_fits)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def handler(body):
    """Handle POST /calibration/photometry and translate service results to HTTP
    responses."""

    request_id = uuid4().hex[:12]
    image_url = body.get("wcs_image_url")

    try:
        stage = "validate_request"
        config = validate(body)

        stage = "cache_or_run_pipeline"
        results = get_or_compute(
            "calibration-photometry",
            config,
            lambda: _run_photometry_uncached(config),
        )

        stage = "build_response"
        results["request_id"] = request_id
        return results, 200, {"Content-Type": "application/json"}
    except InputValidationError as exc:
        payload = {
            "status": "bad_request",
            "message": str(exc),
            "request_id": request_id,
            "stage": stage,
            "image_url": image_url,
        }
        return json_response(payload, 400)
    except BadRequest as exc:
        payload = {
            "status": "bad_request",
            "message": exc.description,
            "request_id": request_id,
            "stage": stage,
            "image_url": image_url,
        }
        return json_response(payload, 400)
    except PhotometricCalibrationError as exc:
        logger.warning(
            "Photometric calibration failed: %s [request_id=%s stage=%s image_url=%r]",
            str(exc),
            request_id,
            stage,
            image_url,
        )
        payload = {
            "status": "calibration_failed",
            "message": str(exc),
            "error_type": type(exc).__name__,
            "request_id": request_id,
            "stage": stage,
            "image_url": image_url,
        }
        return json_response(payload, 422)
    except GeneratedArtifactUploadError as exc:
        logger.exception(
            "Photometry artifact upload failed [request_id=%s stage=%s image_url=%r]",
            request_id,
            stage,
            image_url,
        )
        payload = {
            "status": "artifact_upload_failed",
            "message": str(exc),
            "request_id": request_id,
            "stage": stage,
            "image_url": image_url,
        }
        return json_response(payload, 500)
    except Exception as exc:
        logger.exception(
            "Photometric calibration request failed "
            "[request_id=%s stage=%s image_url=%r]",
            request_id,
            stage,
            image_url,
        )
        payload = {
            "status": "error",
            "message": str(exc),
            "error_type": type(exc).__name__,
            "request_id": request_id,
            "stage": stage,
            "image_url": image_url,
        }
        return json_response(payload, 500)
