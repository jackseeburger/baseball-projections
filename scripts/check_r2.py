import os
"""Check what's in R2."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.r2 import get_s3_client  # noqa: E402

s3 = get_s3_client()

response = s3.list_objects_v2(Bucket='baseball-data')
total = 0
for obj in response.get('Contents', []):
    mb = obj['Size'] / 1e6
    total += obj['Size']
    print(f"{obj['Key']}: {mb:.1f} MB")
if total == 0:
    print("Bucket is empty — nothing uploaded yet")
else:
    print(f"\nTotal: {total / 1e9:.2f} GB, Files: {response.get('KeyCount', 0)}")
