from google.cloud import storage
import os, sys
import shutil

bucket_name = 'cloud_computing_edge3'

# temp directory
path = "./tmp"

# Explicitly use service account credentials by specifying the private key file.
storage_client = storage.Client.from_service_account_json('orbital-age-353410-4de0dfaeda15.json')

bucket = storage_client.bucket(bucket_name)

# Note: Client.list_blobs requires at least package version 1.17.0.
blobs = storage_client.list_blobs(bucket_name)


os.mkdir(path);

for blob in blobs:
    print(blob.name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(blob.name)
    blob.download_to_filename(path + '/' + blob.name)

    print(
        "Downloaded storage object {} from bucket {} to local file {}.".format(
            blob.name, bucket_name, path + '/' + blob.name
        )
    )
    
shutil.rmtree(path)