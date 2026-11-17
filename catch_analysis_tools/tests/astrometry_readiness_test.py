import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ..handlers.calibration import astrometry as astrometry_handler
from ..services.astrometry_readiness import download_index_files as download_module
from ..services.astrometry_readiness import index_source, state
from ..services.astrometry_readiness.index_manifest import (
    MANIFEST_FILENAME,
    build_manifest,
    file_matches_entry,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from ..services.astrometry_readiness.prepare_astrometry_data import (
    prepare_astrometry_data,
)


@pytest.fixture(autouse=True)
def restore_readiness_state():
    original_status = dict(state.status)
    original_worker = state.worker
    yield
    state.status.clear()
    state.status.update(original_status)
    state.worker = original_worker


def test_manifest_inventories_and_verifies_local_files(tmp_path):
    index_file = tmp_path / "index-5204-00.fits"
    index_file.write_bytes(b"canonical-index-data")

    manifest = build_manifest(tmp_path)
    manifest_path = tmp_path / MANIFEST_FILENAME
    write_manifest(manifest, manifest_path)
    loaded = load_manifest(manifest_path)

    assert loaded["files"][0]["name"] == index_file.name
    assert loaded["files"][0]["size"] == index_file.stat().st_size
    assert file_matches_entry(tmp_path, loaded["files"][0], verify_checksum=True)


def test_local_manifest_marks_astrometry_ready_without_s3(tmp_path, monkeypatch):
    index_file = tmp_path / "index-5204-00.fits"
    index_file.write_bytes(b"local-index")
    write_manifest(build_manifest(tmp_path), tmp_path / MANIFEST_FILENAME)
    monkeypatch.setenv("ASTROMETRY_INDEX_DIR", str(tmp_path))
    monkeypatch.delenv("CAT_ASTROMETRY_INDEX_BUCKET", raising=False)
    monkeypatch.delenv("TF_VAR_CAT_ASTROMETRY_INDEX_BUCKET_NAME", raising=False)

    with patch.object(index_source, "get_s3_client") as s3_client:
        result = prepare_astrometry_data()

    s3_client.assert_not_called()
    assert result["ready"] is True
    assert result["state"] == "ready"
    assert result["files_present"] == 1
    assert result["index_source"].startswith("file://")


def test_missing_manifest_records_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("ASTROMETRY_INDEX_DIR", str(tmp_path))
    monkeypatch.delenv("CAT_ASTROMETRY_INDEX_BUCKET", raising=False)
    monkeypatch.delenv("TF_VAR_CAT_ASTROMETRY_INDEX_BUCKET_NAME", raising=False)

    with pytest.raises(RuntimeError, match="Astrometry manifest is missing"):
        prepare_astrometry_data()

    assert state.status["state"] == "error"
    assert state.status["ready"] is False
    assert "--manifest-only" in state.status["error"]


def test_empty_remote_storage_is_seeded_from_private_s3(tmp_path, monkeypatch):
    index_bytes = b"remote-index-data"
    manifest = validate_manifest(
        {
            "schema_version": 1,
            "files": [
                {
                    "name": "index-5204-00.fits",
                    "size": len(index_bytes),
                    "sha256": hashlib.sha256(index_bytes).hexdigest(),
                }
            ],
        }
    )
    objects = {
        f"reference/{MANIFEST_FILENAME}": json.dumps(manifest).encode("utf-8"),
        "reference/index-5204-00.fits": index_bytes,
    }

    class FakeS3Client:
        def __init__(self):
            self.downloads = []

        def download_file(self, bucket, key, filename):
            self.downloads.append((bucket, key))
            Path(filename).write_bytes(objects[key])

    client = FakeS3Client()
    stale_download = tmp_path / ".index-5204-00.fits.abandoned.tmp"
    stale_download.write_bytes(b"partial-download")
    monkeypatch.setenv("ASTROMETRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("CAT_ASTROMETRY_INDEX_BUCKET", "reference-bucket")
    monkeypatch.setenv("CAT_ASTROMETRY_INDEX_PREFIX", "reference/")

    with (
        patch.object(index_source, "get_s3_client", return_value=client),
        patch.object(download_module, "get_s3_client", return_value=client),
    ):
        result = prepare_astrometry_data()

    assert result["ready"] is True
    assert (tmp_path / "index-5204-00.fits").read_bytes() == index_bytes
    assert not stale_download.exists()
    assert client.downloads == [
        ("reference-bucket", f"reference/{MANIFEST_FILENAME}"),
        ("reference-bucket", "reference/index-5204-00.fits"),
    ]


def test_corrupt_s3_index_is_rejected_before_publication(tmp_path, monkeypatch):
    expected_bytes = b"expected-index"
    corrupt_bytes = b"corrupt-index!"
    manifest = validate_manifest(
        {
            "schema_version": 1,
            "files": [
                {
                    "name": "index-5204-00.fits",
                    "size": len(expected_bytes),
                    "sha256": hashlib.sha256(expected_bytes).hexdigest(),
                }
            ],
        }
    )
    objects = {
        f"reference/{MANIFEST_FILENAME}": json.dumps(manifest).encode("utf-8"),
        "reference/index-5204-00.fits": corrupt_bytes,
    }

    class FakeS3Client:
        def download_file(self, bucket, key, filename):
            Path(filename).write_bytes(objects[key])

    client = FakeS3Client()
    monkeypatch.setenv("ASTROMETRY_INDEX_DIR", str(tmp_path))
    monkeypatch.setenv("CAT_ASTROMETRY_INDEX_BUCKET", "reference-bucket")
    monkeypatch.setenv("CAT_ASTROMETRY_INDEX_PREFIX", "reference/")

    with (
        patch.object(index_source, "get_s3_client", return_value=client),
        patch.object(download_module, "get_s3_client", return_value=client),
        pytest.raises(RuntimeError, match="checksum does not match"),
    ):
        prepare_astrometry_data()

    assert not (tmp_path / "index-5204-00.fits").exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert state.status["state"] == "error"
    assert state.status["ready"] is False


def test_calibration_astrometry_waits_for_reference_data():
    readiness = {"state": "downloading", "ready": False, "files_present": 2}

    with (
        patch.object(astrometry_handler, "is_astrometry_ready", return_value=False),
        patch.object(
            astrometry_handler,
            "get_astrometry_readiness_status",
            return_value=readiness,
        ),
        patch.object(astrometry_handler, "get_or_compute") as get_or_compute,
    ):
        response = astrometry_handler.handler(
            {"image_url": "https://example.org/input.fits"}
        )

    get_or_compute.assert_not_called()
    payload = json.loads(response.get_data(as_text=True))
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "30"
    assert payload["status"] == "not_ready"
    assert payload["astrometry_data"] == readiness
