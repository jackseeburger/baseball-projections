"""Check what's in R2."""
import boto3
from botocore.config import Config

s3 = boto3.client('s3',
    endpoint_url='https://108be5c536e5066d63e944b682eb83e7.r2.cloudflarestorage.com',
    aws_access_key_id='18170dc8b2b4805d3c057f69bb5b8ffb',
    aws_secret_access_key='3b7f08cb995538ce5ce81572036d416a5535b8b582e53f2ac2ad80b9fd9a687a',
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
