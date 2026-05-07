from catch_analysis_tools.app.astrometry_readiness import state
from catch_analysis_tools.app.astrometry_readiness.get_current_time import (
    get_current_time,
)


def set_astrometry_readiness_status(**updates):
    """Update the in-memory astrometry readiness status atomically."""
    with state.state_lock:
        state.status.update(updates)
        state.status["updated_at"] = get_current_time()
        return dict(state.status)
