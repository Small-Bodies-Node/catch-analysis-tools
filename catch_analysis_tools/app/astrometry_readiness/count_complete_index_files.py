import glob
import os

from catch_analysis_tools.app.astrometry_readiness.get_index_dir import get_index_dir


def count_complete_index_files(expected_files=None):
    """Count index files that exist locally and are non-empty."""
    if expected_files is None:
        files = glob.glob(str(get_index_dir() / "index-*.fits"))
    else:
        files = [str(get_index_dir() / filename) for filename in expected_files]
    complete_files = [
        path for path in files if os.path.exists(path) and os.path.getsize(path) > 0
    ]
    return len(complete_files)
