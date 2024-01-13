import os
from dotenv import load_dotenv
import boto3

# Load environment variables from .env
load_dotenv()

def upload_folder_to_s3(local_folder,bucket_name):
    # Retrieve AWS credentials from environment variables
    aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
    aws_region = os.environ.get('AWS_REGION')

    # Create an S3 client
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)

    # Iterate through each file in the local folder
    for root, dirs, files in os.walk(local_folder):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_key = os.path.relpath(local_file_path, local_folder).replace("\\", "/")

            try:
                # Upload the file to S3
                s3.upload_file(local_file_path, bucket_name, s3_key)
            except Exception as e:
                print(f"Error uploading {local_file_path}: {e}")