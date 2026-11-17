import logging

from .prepare_astrometry_data import prepare_astrometry_data

logger = logging.getLogger(__name__)


def background_prepare_astrometry_data(force=False):
    """Run astrometry readiness preparation without crashing the app process."""
    try:
        prepare_astrometry_data(force=force)
    except Exception:
        logger.exception("Background astrometry preparation stopped")
