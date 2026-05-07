from catch_analysis_tools.app.astrometry_readiness import state


def is_astrometry_ready():
    """Return whether the process currently considers astrometry ready."""
    with state.state_lock:
        return bool(state.status["ready"])
