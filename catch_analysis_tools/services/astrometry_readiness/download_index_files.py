import os
from uuid import uuid4

from .count_complete_index_files import count_complete_index_files
from .get_index_dir import get_index_dir
from .index_manifest import file_matches_entry, sha256_file
from .index_source import get_index_bucket, get_index_key, get_s3_client
from .set_astrometry_readiness_status import set_astrometry_readiness_status


def _remove_stale_temporary_downloads(target_dir, filename):
    """Remove leftovers from an interrupted download while holding the lock."""
    for stale_path in target_dir.glob(f".{filename}.*.tmp"):
        stale_path.unlink(missing_ok=True)


def download_index_files(expected_files, verify_existing=False):
    """Seed missing index files from private S3 and verify every download."""
    bucket = get_index_bucket()
    if not bucket:
        raise RuntimeError(
            "Local astrometry indexes are incomplete and "
            "CAT_ASTROMETRY_INDEX_BUCKET is not configured."
        )

    target_dir = get_index_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    client = get_s3_client()

    for entry in expected_files:
        if file_matches_entry(target_dir, entry, verify_checksum=verify_existing):
            continue

        final_path = target_dir / entry["name"]
        _remove_stale_temporary_downloads(target_dir, entry["name"])
        tmp_path = final_path.with_name(f".{final_path.name}.{uuid4().hex}.tmp")
        key = get_index_key(entry["name"])
        set_astrometry_readiness_status(
            state="downloading",
            ready=False,
            message=f"Downloading {entry['name']} from private S3.",
            current_file=entry["name"],
            files_present=count_complete_index_files(expected_files),
            error=None,
        )

        try:
            client.download_file(bucket, key, str(tmp_path))
            if tmp_path.stat().st_size != entry["size"]:
                raise RuntimeError(
                    f"Downloaded size does not match manifest for {entry['name']}"
                )
            if sha256_file(tmp_path) != entry["sha256"]:
                raise RuntimeError(
                    f"Downloaded checksum does not match manifest for {entry['name']}"
                )
            os.replace(tmp_path, final_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    set_astrometry_readiness_status(current_file=None)
