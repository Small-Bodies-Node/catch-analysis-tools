"""Exercise production Connexion configuration without calibration or S3 calls."""

import importlib
from pathlib import Path
from unittest.mock import MagicMock

import connexion
import pytest
import yaml

from ..app import app as app_module
from ..handlers.calibration import astrometry as astrometry_handler
from ..handlers.calibration import photometry as photometry_handler

IMAGE_URL = "https://example.org/input.fits"


@pytest.fixture
def cache_calls(monkeypatch):
    def cached_result(route, config, compute):
        if route == "calibration-photometry":
            return {
                "status": "success",
                "photometry": {"color_index": config["color_term"]},
            }
        return {"status": "success"}

    cache = MagicMock(side_effect=cached_result)
    monkeypatch.setattr(astrometry_handler, "get_or_compute", cache)
    monkeypatch.setattr(photometry_handler, "get_or_compute", cache)
    monkeypatch.setattr(astrometry_handler, "is_astrometry_ready", lambda: True)
    return cache


@pytest.fixture
def api_client(request, monkeypatch):
    original_add_api = connexion.FlaskApp.add_api
    overrides = getattr(request, "param", {})

    def register_api(app, specification, **kwargs):
        # Use the real app's registration and validator settings. Override only
        # the in-memory spec for SSOT checks; never rewrite the source YAML.
        spec = yaml.safe_load(
            (Path(app_module.__file__).parent / "api" / specification).read_text()
        )
        for route, defaults in overrides.items():
            properties = spec["paths"][f"/calibration/{route}"]["post"]["requestBody"][
                "content"
            ]["application/json"]["schema"]["properties"]
            for name, value in defaults.items():
                properties[name]["default"] = value
        # Check the documented null response too, without changing production's
        # response-validation policy.
        kwargs["validate_responses"] = True
        return original_add_api(app, spec, **kwargs)

    try:
        with monkeypatch.context() as app_patch:
            app_patch.setattr(connexion.FlaskApp, "add_api", register_api)
            importlib.reload(app_module)
            with app_module.app.test_client() as client:
                yield client
    finally:
        importlib.reload(app_module)


def test_astrometry_omitted_fields_use_yaml_defaults(api_client, cache_calls):
    response = api_client.post("/calibration/astrometry", json={"image_url": IMAGE_URL})

    assert response.status_code == 200, response.text
    assert cache_calls.call_args.args[1] == {
        "image_url": IMAGE_URL,
        "ra": None,
        "dec": None,
        "use_ra_dec": False,
        "pixel_scale": 2.5,
        "scale_low": None,
        "scale_high": None,
        "search_radius": 2.0,
    }


def test_photometry_omitted_fields_use_yaml_defaults(api_client, cache_calls):
    response = api_client.post(
        "/calibration/photometry", json={"wcs_image_url": IMAGE_URL}
    )

    assert response.status_code == 200, response.text
    assert cache_calls.call_args.args[1] == {
        "wcs_image_url": IMAGE_URL,
        "snr_threshold": 3.0,
        "aperture_radius": 7.0,
        "catalog": "PanSTARRS1",
        "cal_band": "r",
        "color_term": "g-r",
    }


def test_null_color_disables_color_correction_and_matches_response_schema(
    api_client, cache_calls
):
    response = api_client.post(
        "/calibration/photometry",
        json={"wcs_image_url": IMAGE_URL, "color_term": None},
    )

    assert response.status_code == 200, response.text
    assert cache_calls.call_args.args[1]["color_term"] is None
    assert response.json()["photometry"]["color_index"] is None


def test_photometry_explicit_values_and_zero_are_preserved(api_client, cache_calls):
    body = {
        "wcs_image_url": IMAGE_URL,
        "snr_threshold": 0,
        "aperture_radius": 0,
        "catalog": "SkyMapper",
        "cal_band": "i",
        "color_term": "r-i",
    }
    response = api_client.post("/calibration/photometry", json=body)

    assert response.status_code == 200, response.text
    assert cache_calls.call_args.args[1] == body


def test_astrometry_zero_coordinates_and_explicit_values_are_preserved(
    api_client, cache_calls
):
    body = {
        "image_url": IMAGE_URL,
        "ra": 0,
        "dec": 0,
        "use_ra_dec": True,
        "pixel_scale": 4.5,
        "scale_low": None,
        "scale_high": None,
        "search_radius": 1.0,
    }
    response = api_client.post("/calibration/astrometry", json=body)

    assert response.status_code == 200, response.text
    assert cache_calls.call_args.args[1] == body


@pytest.mark.parametrize(
    "route,required,field",
    [
        ("astrometry", "image_url", "use_ra_dec"),
        ("astrometry", "image_url", "pixel_scale"),
        ("astrometry", "image_url", "search_radius"),
        ("photometry", "wcs_image_url", "snr_threshold"),
        ("photometry", "wcs_image_url", "aperture_radius"),
        ("photometry", "wcs_image_url", "catalog"),
        ("photometry", "wcs_image_url", "cal_band"),
    ],
)
def test_null_nonnullable_fields_are_rejected_before_handler_work(
    api_client, cache_calls, route, required, field
):
    response = api_client.post(
        f"/calibration/{route}", json={required: IMAGE_URL, field: None}
    )

    assert response.status_code == 400, response.text
    cache_calls.assert_not_called()


@pytest.mark.parametrize(
    "api_client,route,required,expected",
    [
        (
            {"astrometry": {"pixel_scale": 4.5, "search_radius": 1.0}},
            "astrometry",
            "image_url",
            {"pixel_scale": 4.5, "search_radius": 1.0},
        ),
        (
            {
                "photometry": {
                    "snr_threshold": 6.5,
                    "aperture_radius": 4.0,
                    "catalog": "SkyMapper",
                    "cal_band": "i",
                    "color_term": None,
                }
            },
            "photometry",
            "wcs_image_url",
            {
                "snr_threshold": 6.5,
                "aperture_radius": 4.0,
                "catalog": "SkyMapper",
                "cal_band": "i",
                "color_term": None,
            },
        ),
    ],
    indirect=["api_client"],
)
def test_changing_only_spec_defaults_changes_values_received_by_handlers(
    api_client, cache_calls, route, required, expected
):
    response = api_client.post(f"/calibration/{route}", json={required: IMAGE_URL})

    assert response.status_code == 200, response.text
    config = cache_calls.call_args.args[1]
    assert {name: config[name] for name in expected} == expected


def test_restored_swagger_example_can_be_submitted_directly_to_photometry(
    api_client, cache_calls
):
    spec = api_client.get("/openapi.json").json()
    properties = spec["paths"]["/calibration/photometry"]["post"]["requestBody"][
        "content"
    ]["application/json"]["schema"]["properties"]
    example = properties["wcs_image_url"]["example"]
    response = api_client.post(
        "/calibration/photometry", json={"wcs_image_url": example}
    )

    assert response.status_code == 200, response.text
    assert cache_calls.call_args.args[1]["wcs_image_url"] == example
