import os
import argparse
import subprocess 
from google.cloud import storage
import yaml 

with open('llm_training_config.yaml', 'r') as file:
    data = yaml.safe_load(file)
model_name = data.get('project_name')

output_folder = model_name

def upload_folder_to_gcs(bucket_name, source_folder, destination_folder):
    ''''
    Description: Upload a folder to a GCS bucket
    Input: bucket_name: Name of the GCS bucket
           source_folder: Local path of the folder to upload
           destination_folder: Destination folder in the bucket
    '''
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, _, files in os.walk(source_folder):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, source_folder)
            blob_path = os.path.join(destination_folder, relative_path)

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path)
            print(f'Uploaded {local_path} to gs://{bucket_name}/{blob_path}')

            
upload_folder_to_gcs('autotrain_trainer',output_folder , f'models/{model_name}')

print("Model uploaded to GCS bucket successfully")
