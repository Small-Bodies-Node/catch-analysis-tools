#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

from catch_analysis_tools.services.astrometry_readiness.index_manifest import (
    MANIFEST_FILENAME,
    build_manifest,
    write_manifest,
)


def _clean_prefix(prefix):
    return prefix.strip("/")


def _object_key(prefix, filename):
    return "/".join(part for part in (_clean_prefix(prefix), filename) if part)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Inventory local Astrometry.net indexes and upload them to S3."
    )
    parser.add_argument(
        "--directory",
        default="local-data/astrometry-indexes",
        help="Directory containing index-*.fits files.",
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("CAT_ASTROMETRY_INDEX_BUCKET")
        or os.environ.get("TF_VAR_CAT_ASTROMETRY_INDEX_BUCKET_NAME"),
    )
    parser.add_argument(
        "--prefix",
        default=os.environ.get("CAT_ASTROMETRY_INDEX_PREFIX")
        or os.environ.get("TF_VAR_CAT_ASTROMETRY_INDEX_PREFIX")
        or "astrometry-indexes/",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Build the local manifest without uploading anything.",
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation.")
    return parser.parse_args()


def _matches_remote(client, bucket, key, entry):
    try:
        response = client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        error_code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if error_code in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise
    return (
        response.get("ContentLength") == entry["size"]
        and response.get("Metadata", {}).get("sha256") == entry["sha256"]
    )


def main():
    args = _parse_args()
    directory = Path(args.directory).resolve()
    manifest_path = directory / MANIFEST_FILENAME

    print(f"Inventorying {directory}")
    manifest = build_manifest(directory)
    write_manifest(manifest, manifest_path)
    total_bytes = sum(entry["size"] for entry in manifest["files"])
    print(
        f"Manifest written: {manifest_path} "
        f"({len(manifest['files'])} files, {total_bytes / 1024**3:.2f} GiB)"
    )

    if args.manifest_only:
        return 0
    if not args.bucket:
        print(
            "CAT_ASTROMETRY_INDEX_BUCKET or "
            "TF_VAR_CAT_ASTROMETRY_INDEX_BUCKET_NAME must be set.",
            file=sys.stderr,
        )
        return 2

    destination = f"s3://{args.bucket}/{_clean_prefix(args.prefix)}/"
    if not args.yes:
        answer = input(f"Upload this immutable reference set to {destination}? [y/N] ")
        if answer.lower() not in {"y", "yes"}:
            print("Upload cancelled.")
            return 0

    import boto3

    client = boto3.client("s3")
    uploaded = 0
    skipped = 0
    for number, entry in enumerate(manifest["files"], start=1):
        key = _object_key(args.prefix, entry["name"])
        if _matches_remote(client, args.bucket, key, entry):
            skipped += 1
            print(
                f"[{number}/{len(manifest['files'])}] already uploaded: {entry['name']}"
            )
            continue

        print(f"[{number}/{len(manifest['files'])}] uploading: {entry['name']}")
        client.upload_file(
            str(directory / entry["name"]),
            args.bucket,
            key,
            ExtraArgs={
                "ContentType": "application/fits",
                "Metadata": {"sha256": entry["sha256"]},
            },
        )
        if not _matches_remote(client, args.bucket, key, entry):
            raise RuntimeError(f"S3 verification failed for {entry['name']}")
        uploaded += 1

    manifest_key = _object_key(args.prefix, MANIFEST_FILENAME)
    client.upload_file(
        str(manifest_path),
        args.bucket,
        manifest_key,
        ExtraArgs={"ContentType": "application/json"},
    )
    print(
        f"Upload complete: {uploaded} uploaded, {skipped} unchanged; "
        f"manifest at s3://{args.bucket}/{manifest_key}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
