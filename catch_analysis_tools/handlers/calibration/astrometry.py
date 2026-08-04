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
from ...services.result_cache import get_or_compute

logger = logging.getLogger(__name__)


def json_response(payload, status):
    return Response(json.dumps(payload), status=status, mimetype="application/json")


def validate(body: dict[str, Any]) -> dict[str, Any]:
    """Logical validation of POST data."""

    if body["use_ra_dec"] and None in [body["ra"], body["dec"]]:
        raise InputValidationError("ra and dec are required when use_ra_dec is true")


def _run_astrometry_uncached(body):
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
            body["ra"],
            body["dec"],
            body["use_ra_dec"],
            body["pixel_scale"],
            scale_low=body["scale_low"],
            scale_high=body["scale_high"],
            search_radius=body["search_radius"],
        )
    finally:
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
        validate(body)

        stage = "cache_or_run_pipeline"
        results = get_or_compute(
            "astrometry",
            body,
            lambda: _run_astrometry_uncached(body),
        )

        stage = "run_pipeline"
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
