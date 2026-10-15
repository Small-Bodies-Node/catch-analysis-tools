import json
import logging
import os
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4

import requests
from flask import Response
from werkzeug.exceptions import BadRequest

from ...exceptions import AstrometricCalibrationError, InputValidationError
from ...services.astrometry_readiness.get_astrometry_readiness_status import (
    get_astrometry_readiness_status,
)
from ...services.astrometry_readiness.is_astrometry_ready import is_astrometry_ready
from ...services.calibration.astrometry import run_pipeline
from ...services.generated_images import (
    GeneratedArtifactUploadError,
    upload_generated_file,
)
from ...services.publish_preview import publish_preview
from ...services.result_cache import get_or_compute

logger = logging.getLogger(__name__)


def json_response(payload, status):
    return Response(json.dumps(payload), status=status, mimetype="application/json")


def validate(body: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize inputs before building the cache key."""
    image_url = body.get("image_url")
    if not image_url:
        raise InputValidationError("image_url is required")

    config = {
        "image_url": image_url,
        "ra": body.get("ra"),
        "dec": body.get("dec"),
        "use_ra_dec": body.get("use_ra_dec", False),
        "pixel_scale": body.get("pixel_scale", 2.5),
        "scale_low": body.get("scale_low"),
        "scale_high": body.get("scale_high"),
        "search_radius": body.get("search_radius", 2.0),
    }

    if config["use_ra_dec"] and None in [config["ra"], config["dec"]]:
        raise InputValidationError("ra and dec are required when use_ra_dec is true")

    if (
        config["scale_low"] is not None
        and config["scale_high"] is not None
        and config["scale_low"] > config["scale_high"]
    ):
        raise InputValidationError("scale_low must be less than or equal to scale_high")

    return config


def _run_astrometry_uncached(config):
    try:
        response = requests.get(config["image_url"], timeout=60)
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
            config["ra"],
            config["dec"],
            config["use_ra_dec"],
            config["pixel_scale"],
            scale_low=config["scale_low"],
            scale_high=config["scale_high"],
            search_radius=config["search_radius"],
        )
        output_fits = result["output_fits"]

        # Published FITS files survive task replacement and can feed photometry.
        uploaded = upload_generated_file(
            output_fits,
            "calibration-astrometry",
            "astrometry.fits",
            "application/fits",
        )
        return {
            **publish_preview(output_fits, "calibration-astrometry"),
            "status": "success",
            "wcs_image_url": uploaded["url"],
            "wcs_image_s3_key": uploaded["key"],
            "center_ra_deg": result["center_ra_deg"],
            "center_dec_deg": result["center_dec_deg"],
            "pixel_scale": result["pixel_scale"],
        }
    finally:
        if output_fits and os.path.exists(output_fits):
            os.remove(output_fits)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def handler(body):
    """Handle POST /calibration/astrometry and translate service results to HTTP
    responses."""

    if not is_astrometry_ready():
        payload = {
            "status": "not_ready",
            "message": "Astrometry index files are not ready yet.",
            "astrometry_data": get_astrometry_readiness_status(),
        }
        return Response(
            json.dumps(payload),
            status=503,
            mimetype="application/json",
            headers={"Retry-After": "30"},
        )

    request_id = uuid4().hex[:12]
    image_url = body.get("image_url")

    try:
        stage = "validate_request"
        config = validate(body)

        stage = "cache_or_run_pipeline"
        results = get_or_compute(
            "calibration-astrometry",
            config,
            lambda: _run_astrometry_uncached(config),
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
    except AstrometricCalibrationError as exc:
        logger.warning(
            "Astrometric calibration failed: %s [request_id=%s stage=%s image_url=%r]",
            str(exc),
            request_id,
            stage,
            image_url,
        )
        payload = {
            "status": "solve_failed",
            "message": str(exc),
            "error_type": type(exc).__name__,
            "request_id": request_id,
            "stage": stage,
            "image_url": image_url,
        }
        return json_response(payload, 422)
    except GeneratedArtifactUploadError as exc:
        logger.exception(
            "Astrometry FITS upload failed [request_id=%s stage=%s image_url=%r]",
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
            "Astrometry request failed [request_id=%s stage=%s image_url=%r]",
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
