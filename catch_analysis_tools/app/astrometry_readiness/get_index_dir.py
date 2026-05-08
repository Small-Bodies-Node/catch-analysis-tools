import os
from pathlib import Path

from catch_analysis_tools.app.astrometry_readiness.constants import DEFAULT_INDEX_DIR


def get_index_dir():
    """Return the directory where Astrometry.net index files should live."""
    return Path(os.environ.get("ASTROMETRY_INDEX_DIR", DEFAULT_INDEX_DIR))
