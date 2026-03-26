"""Upload local parquet data to Modal Volume.

Usage: modal run modal_functions/upload_data.py
"""
import os
from pathlib import Path
from common import app, data_volume


LOCAL_PARQUET_DIR = Path(__file__).parent.parent / "data" / "parquet"


@app.local_entrypoint()
def main():
    """Upload all parquet files from local data/parquet/ to the Modal volume."""
    if not LOCAL_PARQUET_DIR.exists():
        print(f"❌ Local parquet dir not found: {LOCAL_PARQUET_DIR}")
        return

    files_to_upload: list[tuple[str, bytes]] = []
    for root, _dirs, files in os.walk(LOCAL_PARQUET_DIR):
        for f in files:
            if f.endswith(".parquet"):
                local_path = Path(root) / f
                # Preserve directory structure under /data/parquet/
                rel = local_path.relative_to(LOCAL_PARQUET_DIR)
                remote_path = f"/data/parquet/{rel}"
                files_to_upload.append((str(local_path), remote_path))

    print(f"📦 Found {len(files_to_upload)} parquet files to upload")

    with data_volume.batch_upload() as batch:
        for local_path, remote_path in files_to_upload:
            batch.put_file(local_path, remote_path)
            print(f"  ↑ {remote_path}")

    print(f"✅ Uploaded {len(files_to_upload)} files to 'baseball-data' volume")
