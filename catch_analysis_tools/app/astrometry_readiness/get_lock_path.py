from catch_analysis_tools.app.astrometry_readiness.get_index_dir import get_index_dir


def get_lock_path():
    """Return the cross-process lock path used during index downloads."""
    return get_index_dir().parent / ".index-download.lock"
