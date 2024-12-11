import os
import boto3
from botocore.exceptions import NoCredentialsError

# Wasabi S3 credentials and config
AWS_ACCESS_KEY_ID = '8J824EFSZLNXXTRIDCIF'
AWS_SECRET_ACCESS_KEY = 'IwHDQrnL42iE1vo2Mvmez0YSennQXrrQXN2E4VpG'
WASABI_ENDPOINT = 'https://s3.us-east-2.wasabisys.com'
BUCKET_NAME = 'ai-image-editor-webapp'
FOLDER_NAME = 'cn_inpaint_sdxl_multi_channel_v4/checkpoint-26000/controlnet'  # Folder in the S3 bucket
DOWNLOAD_DIR = '/root/photo-background-generation/ckpts/cn_inpaint_sdxl_multi_channel_v4/checkpoint-26000/controlnet'  # Local directory to save files

# Initialize S3 client with Wasabi
s3 = boto3.client(
    's3',
    endpoint_url=WASABI_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def download_folder_from_s3(bucket_name, folder_name, local_dir):
    try:
        # Create local directory if it doesn't exist
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)

        # List all files in the specified folder on Wasabi S3
        objects = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)

        # Check if folder exists and contains files
        if 'Contents' not in objects:
            print(f"Folder {folder_name} does not exist or is empty.")
            return

        # Download each file
        for obj in objects['Contents']:
            file_key = obj['Key']
            local_file_path = os.path.join(local_dir, file_key.replace(folder_name + '/', ''))
            
            # Create local subdirectory if needed
            if not os.path.exists(os.path.dirname(local_file_path)):
                os.makedirs(os.path.dirname(local_file_path))

            # Download file from Wasabi S3
            print(f"Downloading {file_key} to {local_file_path}...")
            s3.download_file(bucket_name, file_key, local_file_path)

        print(f"Folder {folder_name} successfully downloaded to {local_dir}.")

    except NoCredentialsError:
        print("Credentials not available.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    download_folder_from_s3(BUCKET_NAME, FOLDER_NAME, DOWNLOAD_DIR)

