"""Cloudflare R2 access — the one place that reads the `R2_*` variables.

    R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY   from the R2 API-token page
    R2_ENDPOINT_URL                          https://<account>.r2.cloudflarestorage.com
    R2_BUCKET_NAME                           default "baseball-data"

Cloudflare's token page shows the endpoint with the bucket appended
(".../baseball-data"); boto3 would then add the bucket again and every call
fails with NoSuchKey, so the path is stripped here.
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

DEFAULT_BUCKET = "baseball-data"


def endpoint_url(raw: str | None = None) -> str:
    raw = raw if raw is not None else os.environ["R2_ENDPOINT_URL"]
    u = urlparse(raw.strip())
    if not u.scheme or not u.netloc:
        raise ValueError("R2_ENDPOINT_URL must look like https://<account>.r2.cloudflarestorage.com")
    return f"{u.scheme}://{u.netloc}"


def bucket() -> str:
    return os.getenv("R2_BUCKET_NAME") or DEFAULT_BUCKET


def get_s3_client():
    """boto3 S3 client for R2. Raises KeyError naming the missing variable."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url(),
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def duckdb_settings() -> list[str]:
    """SET statements for DuckDB httpfs against R2."""
    host = urlparse(endpoint_url()).netloc
    return [
        f"SET s3_endpoint='{host}'",
        f"SET s3_access_key_id='{os.environ['R2_ACCESS_KEY_ID']}'",
        f"SET s3_secret_access_key='{os.environ['R2_SECRET_ACCESS_KEY']}'",
        "SET s3_region='auto'",
        "SET s3_url_style='path'",
    ]
