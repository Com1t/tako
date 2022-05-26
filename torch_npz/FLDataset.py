import torch
import numpy as np
import pickle
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

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
        
        # transform
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        
        
    def __getitem__(self, index):
        image = self.transforms(self.x_train[index])
        label = self.y_train[index]
        return image, label
    
    def __len__(self):
        return self.x_train.shape[0]
 
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
    
    
    