import fcntl
import glob
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

INDEX_URL = "https://portal.nersc.gov/project/cosmo/temp/dstn/index-5200/LITE/"
DEFAULT_INDEX_DIR = "/root/.astrometry/data"

_state_lock = threading.RLock()
_worker = None
_status = {
    "state": "unknown",
    "ready": False,
    "message": "Astrometry data has not been checked yet.",
    "files_present": 0,
    "expected_files": None,
    "index_dir": DEFAULT_INDEX_DIR,
    "index_url": INDEX_URL,
    "updated_at": None,
    "error": None,
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def index_dir():
    return Path(os.environ.get("ASTROMETRY_INDEX_DIR", DEFAULT_INDEX_DIR))


def _sentinel_path():
    return index_dir() / ".cat_astrometry_ready.json"


def _lock_path():
    return index_dir().parent / ".index-download.lock"


def _set_status(**updates):
    with _state_lock:
        _status.update(updates)
        _status["updated_at"] = _now()
        return dict(_status)


def get_status():
    with _state_lock:
        status = dict(_status)
    status["index_dir"] = str(index_dir())
    return status


def is_ready():
    with _state_lock:
        return bool(_status["ready"])


def _remote_index_files(session):
    response = session.get(INDEX_URL, timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    files = sorted({
        link["href"]
        for link in soup.find_all("a", href=True)
        if link["href"].startswith("index-") and link["href"].endswith(".fits")
    })
    if not files:
        raise RuntimeError(f"No astrometry index files found at {INDEX_URL}")
    return files


def _count_complete_files(expected_files=None):
    if expected_files is None:
        files = glob.glob(str(index_dir() / "index-*.fits"))
    else:
        files = [str(index_dir() / filename) for filename in expected_files]
    complete_files = [
        path for path in files if os.path.exists(path) and os.path.getsize(path) > 0
    ]
    return len(complete_files)


def index_files_complete(expected_files):
    files_present = _count_complete_files(expected_files)
    return files_present == len(expected_files), files_present


def _write_sentinel(files_present, expected_files):
    sentinel = _sentinel_path()
    payload = {
        "state": "ready",
        "ready": True,
        "files_present": files_present,
        "expected_files": len(expected_files),
        "index_url": INDEX_URL,
        "index_dir": str(index_dir()),
        "completed_at": _now(),
    }
    tmp = sentinel.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, sentinel)


def _download_index_files(session, expected_files):
    target_dir = index_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    for href in expected_files:
        final_path = target_dir / href
        if final_path.exists() and final_path.stat().st_size > 0:
            continue

        tmp_path = final_path.with_name(f".{final_path.name}.tmp")
        file_url = urljoin(INDEX_URL, href)
        _set_status(
            state="downloading",
            ready=False,
            message=f"Downloading {href}",
            files_present=_count_complete_files(expected_files),
            error=None,
        )

        with session.get(file_url, stream=True, timeout=300) as download:
            download.raise_for_status()
            with open(tmp_path, "wb") as handle:
                for chunk in download.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

        os.replace(tmp_path, final_path)


def _acquire_lock():
    lock = _lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock, "w", encoding="utf-8")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def prepare_astrometry_data(force=False):
    target_dir = index_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        expected_files = _remote_index_files(session)
        expected_count = len(expected_files)

        _set_status(
            state="checking",
            ready=False,
            message="Checking astrometry index files.",
            index_dir=str(target_dir),
            index_url=INDEX_URL,
            expected_files=expected_count,
            error=None,
        )

        complete, files_present = index_files_complete(expected_files)
        if complete and not force:
            _write_sentinel(files_present, expected_files)
            _set_status(
                state="ready",
                ready=True,
                message="Astrometry index files are ready.",
                files_present=files_present,
                error=None,
            )
            return get_status()

        lock_handle = _acquire_lock()
        try:
            complete, files_present = index_files_complete(expected_files)
            if complete and not force:
                _write_sentinel(files_present, expected_files)
                _set_status(
                    state="ready",
                    ready=True,
                    message="Astrometry index files are ready.",
                    files_present=files_present,
                    error=None,
                )
                return get_status()

            _set_status(
                state="downloading",
                ready=False,
                message="Astrometry index files are incomplete. Downloading missing files.",
                files_present=files_present,
                error=None,
            )
            _download_index_files(session, expected_files)

            complete, files_present = index_files_complete(expected_files)
            if not complete:
                raise RuntimeError(
                    "Astrometry index files incomplete: "
                    f"{files_present} / {expected_count}"
                )

            _write_sentinel(files_present, expected_files)
            _set_status(
                state="ready",
                ready=True,
                message="Astrometry index files are ready.",
                files_present=files_present,
                error=None,
            )
            return get_status()
        except Exception as exc:
            _set_status(
                state="error",
                ready=False,
                message="Astrometry index preparation failed.",
                files_present=_count_complete_files(expected_files),
                error=str(exc),
            )
            raise
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


def _background_prepare(force=False):
    try:
        prepare_astrometry_data(force=force)
    except Exception:
        pass


def start_background_check(force=False):
    global _worker
    with _state_lock:
        if _worker is not None and _worker.is_alive():
            return dict(_status)

        _status.update({
            "state": "checking",
            "ready": False,
            "message": "Astrometry data readiness check has started.",
            "index_dir": str(index_dir()),
            "index_url": INDEX_URL,
            "expected_files": None,
            "error": None,
            "updated_at": _now(),
        })
        _worker = threading.Thread(
            target=_background_prepare,
            kwargs={"force": force},
            daemon=True,
        )
        _worker.start()
        return dict(_status)


def health():
    return {
        "status": "ok",
        "astrometry_data": get_status(),
    }


def reset():
    status = start_background_check(force=True)
    return {
        "status": "reset_started",
        "astrometry_data": status,
    }, 202
