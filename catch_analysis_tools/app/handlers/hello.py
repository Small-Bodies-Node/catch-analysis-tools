import os

from catch_analysis_tools import __version__


def hello():
    """Return a tiny liveness response with build identity."""
    docker_image_tag = os.environ.get("DOCKER_IMAGE_TAG", "unknown")
    return (
        "Catch Analysis Tools is running.\n"
        f"package_version={__version__}\n"
        f"docker_image_tag={docker_image_tag}\n"
    )
