import fcntl

from catch_analysis_tools.app.astrometry_readiness.get_lock_path import get_lock_path


def acquire_index_download_lock():
    """Acquire the cross-process lock for mutating the shared index dir."""
    lock = get_lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock, "w", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle
