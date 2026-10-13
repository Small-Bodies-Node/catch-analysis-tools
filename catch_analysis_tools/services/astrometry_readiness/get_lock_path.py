from .get_index_dir import get_index_dir


def get_lock_path():
    """Return the cross-process lock path used during index downloads."""
    # Keep the lock on the shared bind mount/EFS so all containers see it.
    return get_index_dir() / ".index-download.lock"
