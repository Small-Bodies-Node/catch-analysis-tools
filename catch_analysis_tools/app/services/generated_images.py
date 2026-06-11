import io
import os
from uuid import uuid4


def _clean_prefix(prefix):
    if not prefix:
        return ""
    return prefix.strip("/")


def _public_url(base_url, key):
    return f"{base_url.rstrip('/')}/{key}"


def upload_figure_png(fig, route_name, image_name):
    """Upload a Matplotlib figure to the public CAT images bucket."""
    import boto3

    bucket = os.environ.get("CAT_IMAGES_BUCKET")
    if not bucket:
        raise RuntimeError("CAT_IMAGES_BUCKET is not set.")

    prefix = _clean_prefix(os.environ.get("CAT_IMAGES_PREFIX", "generated-images/"))
    base_url = os.environ.get("CAT_IMAGES_BASE_URL")
    region = os.environ.get("AWS_REGION")

    key_parts = [part for part in [prefix, route_name, f"{uuid4().hex}-{image_name}"] if part]
    key = "/".join(key_parts)

    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format="png", dpi=150, bbox_inches="tight")
    image_buffer.seek(0)

    client_kwargs = {}
    if region:
        client_kwargs["region_name"] = region

    boto3.client("s3", **client_kwargs).put_object(
        Bucket=bucket,
        Key=key,
        Body=image_buffer.getvalue(),
        ContentType="image/png",
        CacheControl="public, max-age=86400",
    )

    if not base_url:
        if region:
            base_url = f"https://{bucket}.s3.{region}.amazonaws.com"
        else:
            base_url = f"https://{bucket}.s3.amazonaws.com"

    return {
        "url": _public_url(base_url, key),
        "bucket": bucket,
        "key": key,
    }
