import threading

from .get_index_dir import get_index_dir

state_lock = threading.RLock()
worker = None
status = {
    "state": "unknown",
    "ready": False,
    "message": "Astrometry data has not been checked yet.",
    "files_present": 0,
    "expected_files": None,
    "index_dir": str(get_index_dir().absolute()),
    "index_source": None,
    "current_file": None,
    "updated_at": None,
    "error": None,
}
