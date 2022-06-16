import torch
import numpy as np
import pickle
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from google.cloud import storage
import os, sys
import shutil
import errno
import secrets

torch.manual_seed(1)  # reproducible
 
class FLDataset(Dataset):
    def __init__(self, datapath, purpose):
        # load data
        with open(datapath, 'rb') as fp:
            (x_train, y_train, x_test, y_test) = pickle.load(fp)
            if(purpose == 'train'):
                self.x_train = np.expand_dims(x_train.astype('float32'), axis=-1)
                self.y_train = y_train.astype('int64')
            else:
                self.x_train = np.expand_dims(x_test.astype('float32'), axis=-1)
                self.y_train = y_test.astype('int64')
        
        self.hash = secrets.token_urlsafe(16)
        
        # temp directory
        self.path = "./tmp" + self.hash
        
        self.find_cloud_data()
        
        # transform
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        
        
    def __getitem__(self, index):
        image = self.transforms(self.x_train[index])
        label = self.y_train[index]
        return image, label
    
    def find_cloud_data(self):

        bucket_name = 'cloud_computing_edge3'


        # Explicitly use service account credentials by specifying the private key file.
        storage_client = storage.Client.from_service_account_json('orbital-age-353410-4de0dfaeda15.json')

        bucket = storage_client.bucket(bucket_name)

        # Note: Client.list_blobs requires at least package version 1.17.0.
        blobs = storage_client.list_blobs(bucket_name)

        try:
            os.mkdir(self.path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
            pass
        
        file_names = []
        for blob in blobs:
            print(blob.name)

            # Construct a client side representation of a blob.
            # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
            # any content from Google Cloud Storage. As we don't need additional data,
            # using `Bucket.blob` is preferred here.
            blob = bucket.blob(blob.name)
            blob.download_to_filename(self.path + '/' + blob.name)
            file_names.append(self.path + '/' + blob.name)
            print(
                "Downloaded storage object {} from bucket {} to local file {}.".format(
                    blob.name, bucket_name, self.path + '/' + blob.name
                )
            )
            
        
        for file in file_names:
            train_data = np.load(file, allow_pickle=True)
            x_train = np.expand_dims(train_data['x_train'].astype('float32'), axis=-1)
            y_train = train_data['y_train']
            print(self.x_train.shape)
            print(self.y_train.shape)
            print(x_train.shape)
            print(y_train.shape)
            self.x_train = np.concatenate((self.x_train, x_train), axis=0)
            self.y_train = np.concatenate((self.y_train, y_train), axis=0)
            
        
    def __len__(self):
        return self.x_train.shape[0]
    
    def __del__(self):
        shutil.rmtree(self.path, ignore_errors=True)
 
def main():
    dataset = FLDataset('/ML/FL_algo/IIDdataset/client_1.pickle', 'train')
    data = DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True)
    for i, (batch) in enumerate(data):
        print(i)
        print(batch[0].dtype)
        print(batch[1].dtype)
        i = i
 
if __name__ == '__main__':
    main()
    
    
    