import zmq
import threading
import pandas as pd
import numpy as np
import time


'''
Model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dset
import torch.optim as optim
import time

import torchvision
from torchvision import datasets, transforms

torch.manual_seed(1)


class MNIST_NN(nn.Module):
    def __init__(self):
        super(MNIST_NN, self).__init__()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=(5,5))
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=(5,5))
        
        self.linear_1 = nn.Linear(1024, 256)
        self.linear_2 = nn.Linear(256, 10)
        
    def forward(self, image):
        out = F.max_pool2d(F.relu(self.conv_1(image)), kernel_size=(2,2))
        out = F.max_pool2d(F.relu(self.conv_2(out)), kernel_size=(2,2))
        
        out = self.linear_1(torch.flatten(out, 1))
        out = self.linear_2(out)
        
        return out
    
'''
Global model Initialize
'''
global_model = MNIST_NN()

# client database
# dtypes = np.dtype(
#     [
#         ("ip", str),
#         ("port", int),
#         ("status", str),
#         ("FPS", np.datetime64),
#     ]
# )
# df = pd.DataFrame(np.empty(0, dtype=dtypes))

# FL hyperparameter
active_client_num = 10
round_num = 20
C = 0.5
E = 1

# client info (for threads)
client_infos = []

# reply to server
model_queue = []

class client(threading.Thread):
    def __init__(self, client_num, global_model_state_dict, E):
        threading.Thread.__init__(self)
        # client_num
        self.client_num = client_num
        
        # model training
        self.E = E
        
        # local model
        self.device = f'cuda:{self.client_num % 4}' if torch.cuda.is_available() else 'cpu'
        self.model = MNIST_NN().to(self.device)
        
        # load global model
        self.model.load_state_dict(global_model_state_dict)
        
        # data loading
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,)),]
        )
        from torch_npz.FLDataset import FLDataset
        self.train_data = FLDataset(f'/ML/FL_algo/IIDdataset/client_{self.client_num}.pickle', 'train')
        self.trainLoader = dset.DataLoader(self.train_data, batch_size=256, shuffle=True)
        
        # optimizer
        self.loss_function = nn.CrossEntropyLoss()
        self.loss = None
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # system info
        self.FPS = 10
        self.data_amt = 10
        self.gradient_norm = 100
        
    def run(self):
        global model_queue
        
        for epoch in range(self.E):
            for i, (data) in enumerate(self.trainLoader):
                self.model.zero_grad()

                image = data[0].to(self.device)
                labels = data[1].to(self.device)

                res = self.model(image)

                self.loss = self.loss_function(res, labels)

                if (i + 1) % 100 == 0:
                    print('Client: {} Epoch [{}/{}], Step [{}], Loss: {:.4f}'.format(self.client_num, epoch + 1, self.E, i + 1, self.loss.item()))

                self.loss.backward()
                self.optimizer.step()

        model_queue.append({'sys_info': {"FPS": self.FPS, 
                                         "data_amt": self.data_amt,
                                         "gradient_norm": self.gradient_norm},
                            'model': self.model.state_dict()})
        
'''
Initialize
'''
        
'''
Training
'''
from torch_npz.FLDataset import FLDataset
test_data = FLDataset('/ML/FL_algo/IIDdataset/client_1.pickle', 'test')
testLoader = dset.DataLoader(test_data, batch_size=1024, shuffle=False)

for round in range(round_num):
    print(f'Round {round} started!')
    for client_num in range(active_client_num):
        client_infos.append(client(client_num, global_model.state_dict(), E))
        client_infos[client_num].start()
    
    while(len(model_queue) != active_client_num):
        time.sleep(0.1)
    
    print("client side complete!")
    
    while len(client_infos) > 0:
        temp_info = client_infos.pop()
        temp_info.join()
        del temp_info
        
    client_infos.clear()

    # 'Aggregation'
    model_dict = global_model.state_dict()
    for k in model_dict.keys():
        model_dict[k] = torch.stack([model_queue[i]['model'][k].to('cpu').float() for i in range(len(model_queue))], 0).mean(0)
    
    model_queue.clear()
    
    global_model.load_state_dict(model_dict)
    
    with torch.no_grad():
        correct_count = 0
        for _, (data) in enumerate(testLoader):
            image = data[0]
            labels = data[1]
            output = global_model(image)
            predict_label = torch.argmax(nn.Softmax(dim=1)(output), dim=1, keepdim=False)
            correct_count += (predict_label == labels).float().sum()
        acc = correct_count / len(test_data)
        print('Acc : {:.4f}'.format(acc.item()))
    
    print(f'Round {round} completed!')