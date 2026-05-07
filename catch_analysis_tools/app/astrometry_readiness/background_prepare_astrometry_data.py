from catch_analysis_tools.app.astrometry_readiness.prepare_astrometry_data import (
    prepare_astrometry_data,
)


def background_prepare_astrometry_data(force=False):
    """Run astrometry readiness preparation without crashing the app process."""
    try:
        prepare_astrometry_data(force=force)
    except Exception:
        pass
