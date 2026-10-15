import fcntl
import logging

from .acquire_index_download_lock import acquire_index_download_lock
from .count_complete_index_files import count_complete_index_files
from .download_index_files import download_index_files
from .get_astrometry_readiness_status import get_astrometry_readiness_status
from .get_index_dir import get_index_dir
from .index_files_complete import index_files_complete
from .index_source import load_or_download_manifest
from .set_astrometry_readiness_status import set_astrometry_readiness_status
from .write_ready_sentinel import write_ready_sentinel

logger = logging.getLogger(__name__)


def prepare_astrometry_data(force=False):
    """Check/download index files and mark astrometry ready when complete."""
    target_dir = get_index_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    expected_files = []
    index_source = None

    try:
        manifest, index_source = load_or_download_manifest()
        expected_files = manifest["files"]
        expected_count = len(expected_files)
        set_astrometry_readiness_status(
            state="checking",
            ready=False,
            message="Checking astrometry index files.",
            index_dir=str(target_dir),
            index_source=index_source,
            expected_files=expected_count,
            current_file=None,
            error=None,
        )

        complete, files_present = index_files_complete(
            expected_files, verify_checksum=force
        )
        if complete and not force:
            write_ready_sentinel(files_present, expected_files, index_source)
            set_astrometry_readiness_status(
                state="ready",
                ready=True,
                message="Astrometry index files are ready.",
                files_present=files_present,
                current_file=None,
                error=None,
            )
            return get_astrometry_readiness_status()

        lock_handle = acquire_index_download_lock()
        try:
            complete, files_present = index_files_complete(
                expected_files, verify_checksum=force
            )
            if complete:
                write_ready_sentinel(files_present, expected_files, index_source)
                set_astrometry_readiness_status(
                    state="ready",
                    ready=True,
                    message="Astrometry index files are ready.",
                    files_present=files_present,
                    current_file=None,
                    error=None,
                )
                return get_astrometry_readiness_status()

            set_astrometry_readiness_status(
                state="downloading",
                ready=False,
                message="Downloading missing astrometry indexes from private S3.",
                files_present=files_present,
                current_file=None,
                error=None,
            )
            download_index_files(expected_files, verify_existing=force)

            complete, files_present = index_files_complete(expected_files)
            if not complete:
                raise RuntimeError(
                    "Astrometry index files incomplete: "
                    f"{files_present} / {expected_count}"
                )

            write_ready_sentinel(files_present, expected_files, index_source)
            set_astrometry_readiness_status(
                state="ready",
                ready=True,
                message="Astrometry index files are ready.",
                files_present=files_present,
                current_file=None,
                error=None,
            )
            return get_astrometry_readiness_status()
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
    except Exception as exc:
        logger.exception("Astrometry index preparation failed")
        set_astrometry_readiness_status(
            state="error",
            ready=False,
            message="Astrometry index preparation failed.",
            index_source=index_source,
            expected_files=len(expected_files) if expected_files else None,
            files_present=count_complete_index_files(expected_files or None),
            current_file=None,
            error=str(exc),
        )
        raise
