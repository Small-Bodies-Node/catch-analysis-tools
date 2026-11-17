import os
from pathlib import Path
from uuid import uuid4

from ..generated_images import _get_aws_client_kwargs
from .constants import DEFAULT_INDEX_PREFIX
from .get_index_dir import get_index_dir
from .index_manifest import MANIFEST_FILENAME, load_manifest


def _clean_prefix(prefix):
    return prefix.strip("/")


def get_manifest_path():
    """Return the manifest path inside the mounted index directory."""
    configured = os.environ.get("ASTROMETRY_INDEX_MANIFEST")
    return Path(configured) if configured else get_index_dir() / MANIFEST_FILENAME


def get_index_bucket():
    """Return the private S3 bucket used to seed remote index storage."""
    return os.environ.get("CAT_ASTROMETRY_INDEX_BUCKET") or os.environ.get(
        "TF_VAR_CAT_ASTROMETRY_INDEX_BUCKET_NAME"
    )


def get_index_prefix():
    return _clean_prefix(
        os.environ.get("CAT_ASTROMETRY_INDEX_PREFIX")
        or os.environ.get("TF_VAR_CAT_ASTROMETRY_INDEX_PREFIX")
        or DEFAULT_INDEX_PREFIX
    )


def get_index_key(filename):
    return "/".join(part for part in (get_index_prefix(), filename) if part)


def get_s3_source():
    bucket = get_index_bucket()
    if not bucket:
        return None
    return f"s3://{bucket}/{get_index_prefix()}/"


def get_s3_client():
    import boto3

    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or os.environ.get("TF_VAR_AWS_REGION")
        or "us-east-1"
    )
    return boto3.client("s3", **_get_aws_client_kwargs(region))


def load_or_download_manifest():
    """Use a mounted manifest, or seed it from the canonical private bucket."""
    manifest_path = get_manifest_path()
    if manifest_path.is_file():
        return load_manifest(manifest_path), f"file://{manifest_path}"

    bucket = get_index_bucket()
    if not bucket:
        raise RuntimeError(
            f"Astrometry manifest is missing at {manifest_path}. Copy the index "
            "files there and run ./_upload_astrometry_indexes --manifest-only, "
            "or configure CAT_ASTROMETRY_INDEX_BUCKET."
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(f".{manifest_path.name}.{uuid4().hex}.tmp")
    key = get_index_key(MANIFEST_FILENAME)
    try:
        get_s3_client().download_file(bucket, key, str(temporary))
        manifest = load_manifest(temporary)
        temporary.replace(manifest_path)
        return manifest, f"s3://{bucket}/{key}"
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download astrometry manifest from s3://{bucket}/{key}"
        ) from exc
