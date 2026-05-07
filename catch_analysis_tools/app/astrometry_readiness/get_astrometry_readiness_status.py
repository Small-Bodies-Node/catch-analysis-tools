from catch_analysis_tools.app.astrometry_readiness import state
from catch_analysis_tools.app.astrometry_readiness.get_index_dir import get_index_dir


def get_astrometry_readiness_status():
    """Return a snapshot of the current astrometry index readiness state."""
    with state.state_lock:
        status = dict(state.status)
    status["index_dir"] = str(get_index_dir())
    return status
