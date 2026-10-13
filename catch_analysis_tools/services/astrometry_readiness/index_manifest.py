import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_FILENAME = "astrometry-index-manifest.json"
MANIFEST_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class AstrometryIndexManifestError(RuntimeError):
    """Raised when the index manifest is missing or unsafe."""


def sha256_file(path, chunk_size=1024 * 1024):
    """Hash a large FITS file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(payload):
    """Validate and normalize the small manifest that describes index files."""
    if not isinstance(payload, dict):
        raise AstrometryIndexManifestError("Astrometry index manifest must be JSON.")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise AstrometryIndexManifestError(
            f"Unsupported astrometry manifest schema: {payload.get('schema_version')}"
        )

    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise AstrometryIndexManifestError(
            "Astrometry index manifest does not contain any files."
        )

    normalized_files = []
    seen_names = set()
    for item in files:
        if not isinstance(item, dict):
            raise AstrometryIndexManifestError("Manifest file entries must be objects.")
        name = item.get("name")
        size = item.get("size")
        checksum = item.get("sha256")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.startswith("index-")
            or not name.endswith(".fits")
        ):
            raise AstrometryIndexManifestError(f"Unsafe index filename: {name!r}")
        if name in seen_names:
            raise AstrometryIndexManifestError(f"Duplicate index filename: {name}")
        if not isinstance(size, int) or size <= 0:
            raise AstrometryIndexManifestError(f"Invalid size for {name}: {size!r}")
        if not isinstance(checksum, str) or not _SHA256_PATTERN.fullmatch(checksum):
            raise AstrometryIndexManifestError(f"Invalid SHA-256 for {name}.")

        seen_names.add(name)
        normalized_files.append({"name": name, "size": size, "sha256": checksum})

    normalized = dict(payload)
    normalized["files"] = sorted(normalized_files, key=lambda item: item["name"])
    return normalized


def load_manifest(path):
    """Read and validate a manifest from disk."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AstrometryIndexManifestError(
            f"Could not read astrometry index manifest: {path}"
        ) from exc
    return validate_manifest(payload)


def build_manifest(index_dir):
    """Inventory local index files for later verification and S3 upload."""
    directory = Path(index_dir)
    files = sorted(directory.glob("index-*.fits"))
    if not files:
        raise AstrometryIndexManifestError(
            f"No index-*.fits files found in {directory}"
        )

    entries = []
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise AstrometryIndexManifestError(
                f"Index files must be regular files, not links: {path}"
            )
        entries.append(
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    return validate_manifest(
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "files": entries,
        }
    )


def write_manifest(payload, path):
    """Write a validated manifest atomically."""
    payload = validate_manifest(payload)
    destination = Path(path)
    temporary = destination.with_suffix(f"{destination.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


def file_matches_entry(index_dir, entry, verify_checksum=False):
    """Return whether one local file matches its manifest entry."""
    path = Path(index_dir) / entry["name"]
    try:
        if not path.is_file() or path.stat().st_size != entry["size"]:
            return False
    except OSError:
        return False

    if verify_checksum:
        return sha256_file(path) == entry["sha256"]
    return True
