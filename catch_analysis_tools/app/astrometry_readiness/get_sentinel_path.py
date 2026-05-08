from catch_analysis_tools.app.astrometry_readiness.get_index_dir import get_index_dir


def get_sentinel_path():
    """Return the marker file path written after index files are ready."""
    return get_index_dir() / ".cat_astrometry_ready.json"
