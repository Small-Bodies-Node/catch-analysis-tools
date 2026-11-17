from .generated_images import upload_generated_bytes
from .render_north_up_preview import render_north_up_preview


def publish_preview(filename: str, route: str) -> dict[str, str]:
    """Publish a preview whose WCS is the same solution as the scientific FITS."""
    uploaded = upload_generated_bytes(
        render_north_up_preview(filename), route, "north-up-preview.png", "image/png"
    )
    return {
        "preview_url": uploaded["url"],
        "preview_s3_key": uploaded["key"],
        "preview_orientation": "north-up-east-left",
        "preview_renderer_version": "north-up-v1",
    }
