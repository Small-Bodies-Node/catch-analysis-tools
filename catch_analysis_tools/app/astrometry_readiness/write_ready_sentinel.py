import json
import os

from catch_analysis_tools.app.astrometry_readiness.constants import INDEX_URL
from catch_analysis_tools.app.astrometry_readiness.get_current_time import (
    get_current_time,
)
from catch_analysis_tools.app.astrometry_readiness.get_index_dir import get_index_dir
from catch_analysis_tools.app.astrometry_readiness.get_sentinel_path import (
    get_sentinel_path,
)


def write_ready_sentinel(files_present, expected_files):
    """Write a small marker file recording that local index files are ready."""
    payload = {
        "state": "ready",
        "ready": True,
        "files_present": files_present,
        "expected_files": len(expected_files),
        "index_url": INDEX_URL,
        "index_dir": str(get_index_dir()),
        "completed_at": get_current_time(),
    }
    sentinel = get_sentinel_path()
    tmp = sentinel.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, sentinel)
