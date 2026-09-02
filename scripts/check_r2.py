import os
"""Check what's in R2."""
import boto3
from botocore.config import Config

s3 = boto3.client('s3',
    endpoint_url=os.environ['R2_ENDPOINT_URL'],
    aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],
    aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],
    config=Config(signature_version='s3v4'),
    region_name='auto'
)

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
