import os
from google.cloud import storage
import nltk
nltk.download('punkt')

# local_source_folder = 'data'
# os.makedirs(local_source_folder, exist_ok=True)


def download_files_from_gcs(bucket_name, local_directory):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs()
    print(blobs)
    os.makedirs(local_directory, exist_ok=True)

    for blob in blobs:
        print(blob.name)
        if blob.name.endswith('.csv') :
            destination_file_name = os.path.join(local_directory, "data",os.path.basename(blob.name))
            blob.download_to_filename(destination_file_name)
            print(f"Downloaded {blob.name} to {destination_file_name}")
            csv_downloaded = True
        elif blob.name.endswith('.yaml') :
            destination_file_name = os.path.join(local_directory, os.path.basename(blob.name))
            blob.download_to_filename(destination_file_name)
            print(f"Downloaded {blob.name} to {destination_file_name}")
            yaml_downloaded = True

        # if csv_downloaded and yaml_downloaded:
        #     break

    if not csv_downloaded:
        print("No .csv file found in the bucket.")
    if not yaml_downloaded:
        print("No .yaml file found in the bucket.")

# Usage
bucket_name = 'autotrain_trainer'
local_directory = './'

download_files_from_gcs(bucket_name, local_directory)

