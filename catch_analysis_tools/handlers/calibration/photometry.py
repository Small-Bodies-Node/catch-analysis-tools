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
from ...services.result_cache import get_or_compute

logger = logging.getLogger(__name__)


def json_response(payload, status):
    return Response(json.dumps(payload), status=status, mimetype="application/json")


def validate(body):
    """Logical validation of POST data."""

    cat = catalogs[body["catalog"]]

    # check that the catalog has the requested bands
    if body["cal_band"] not in cat:
        raise InputValidationError(
            f"Cannot calibrate to {body['cal_band']} with catalog={body['catalog']}"
        )

    color_index = body.get("color_index", None)
    if color_index:
        bands = color_index.split("-")
        if len(bands) != 2:
            raise InputValidationError(
                "Invalid format for color_index: must be band1-band2."
            )

        if bands[0] == bands[1]:
            raise InputValidationError("Invalid color_index: both bands are the same.")

        for band in bands:
            if band not in cat:
                raise InputValidationError(
                    f"Invalid color_index: cannot calibrate to {band} with catalog="
                    + {body["catalog"]}
                )


def _run_photometry_uncached(body):
    try:
        response = requests.get(body["image_url"], timeout=60)
        response.raise_for_status()
    except requests.RequestException:
        raise BadRequest("Could not retrieve FITS image")

    with NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        return run_pipeline(
            tmp_path,
            body["snr_threshold"],
            body["aperture_radius"],
            body["catalog"],
            body["cal_band"],
            color_index=body["color_index"],
            return_plot=body["return_plot"],
            plot_type=body["plot_type"],
        )
    finally:
        os.remove(tmp_path)


def handler(body):
    """Handle POST /calibration/photometry and translate service results to HTTP
    responses."""

    request_id = uuid4().hex[:12]
    image_url = body.get("image_url")

    try:
        stage = "validate_request"
        validate(body)

        stage = "cache_or_run_pipeline"
        results = get_or_compute(
            "photometry",
            body,
            lambda: _run_photometry_uncached(body),
        )

        stage = "build_response"

        if body["return_plot"]:
            plot_type = body["plot_type"]
            if plot_type not in results.get("plots", {}):
                raise BadRequest(f"Unknown plot_type: {plot_type}")

            image_bytes = base64.b64decode(results["plots"][plot_type])
            return Response(image_bytes, mimetype="image/png")

        results["request_id"] = request_id
        results["image_url"] = image_url
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
