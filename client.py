import zmq
import sys
import time
import json
import pandas as pd
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as dset
import torch.optim as optim
import time

import torchvision
from torchvision import datasets, transforms

torch.manual_seed(1)

image_x = 28
image_y = 28
image_channel = 1
output_channel = 10

class MNIST_NN(nn.Module):
    def __init__(self, image_x, image_y, image_channel, output_channel):
        super(MNIST_NN, self).__init__()
        
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=(5,5))
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=(5,5))
        
        self.linear_1 = nn.Linear(1024, 256)
        self.linear_2 = nn.Linear(256, output_channel)
        
    def forward(self, image):
        out = F.max_pool2d(F.relu(self.conv_1(image)), kernel_size=(2,2))
        out = F.max_pool2d(F.relu(self.conv_2(out)), kernel_size=(2,2))
        
        out = self.linear_1(torch.flatten(out, 1))
        out = self.linear_2(out)
        
        return out
    
class fit(threading.Thread):
    def __init__(self, E):
        global model
        threading.Thread.__init__(self)
        self.E = E
        
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,)),]
        )
        self.train_data = datasets.MNIST(root='MNIST', download=True, train=True, transform=self.transform)
        self.trainLoader = dset.DataLoader(self.train_data, batch_size=256, shuffle=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.loss = None
        print(model.parameters())
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    def run(self):
        global training_complete, model, batch_grad_norm, device
        for epoch in range(self.E):
            for i, (data) in enumerate(self.trainLoader):
                model.zero_grad()

                image = data[0].to(device)
                labels = data[1].to(device)

                res = model(image)

                self.loss = self.loss_function(res, labels)

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}], Loss: {:.4f}'.format(epoch + 1, self.E, i + 1, self.loss.item()))

                self.loss.backward()
                self.optimizer.step()
    
        training_complete = True
        
class client(threading.Thread):
    def __init__(self, ip, port, timeout, cpu, data_amt, connectivity):
        threading.Thread.__init__(self)
        self.training = True
        
        # heartbeat
        self.heartrate = 0
        self.timeout = timeout
        
        # system information
        self.cpu = cpu
        self.gradient_norm = 0.0
        self.data_amt = data_amt
        self.connectivity = connectivity
        self.df = pd.DataFrame({
            "ip": ip,
            "port": port,
            "cpu": self.cpu, 
            "gradient_norm": self.gradient_norm,
            "data amount": self.data_amt,
            "connectivity": self.connectivity
        }, index=[0])
        
        # connction related variables
        self.context = zmq.Context()
        self.socket = None
        
        # Estblish Initial Connection
        self.build_init_connect()
        
        # Estblish p2p Connection
        self.build_p2p_connect()
        
        # training module
        self.training_obj = None
        
    def run(self):
        global training_complete, model, batch_grad_norm
        while(self.training):
            # wait a bit
            time.sleep(0.2)
            msg = self.recv()

            if msg is not None:
                # we should only receive model or terminate signal
                # terminate
                if(msg == 'training complete'):
                    self.training = False
                # training on thread in order to keep out heart beating
                elif(type(msg) == dict):
                    print("chosen")
                    epoch = int(msg['E'])
                    model.load_state_dict(msg['model'])
                    # start training thread
                    training_complete = False
                    self.training_obj = fit(epoch)
                    self.training_obj.start()
                else:
                    print(msg, type(msg))

            if(training_complete):
                self.training_obj.join()
                training_complete = False
                self.send({'sys_info': {"cpu": self.cpu, 
                                        "gradient_norm": self.gradient_norm,
                                        "data amount": self.data_amt,
                                        "connectivity": self.connectivity}, 
                           'model': model.state_dict()})
                
            self.heartrate += 1
            if(self.heartrate == 30):
                self.heartrate = 0
                self.send("heartbeat")
                print("heartbeat")
        
    def __del__(self):
        self.socket.close()
        self.context.term()
        
    def build_init_connect(self):
        # Estblish Initial Connection
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://127.0.0.1:7788")
        # send system information
        self.socket.send_pyobj(self.df)
        # wait ack
        print(self.socket.recv_pyobj())
        # close initial connection if server acked
        self.socket.close()
        
    def build_p2p_connect(self):
        # Estblish p2p Connection
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.bind(f"tcp://*:{port}")
        # wait server connects us
        print(self.socket.recv_pyobj())
        # send ack back to server
        self.socket.send_pyobj("client side connection built")
        
    # sync recv
    def send(self, obj):
        self.socket.send_pyobj(obj)
        
    # async recv
    def recv(self):
        try:
            #check for a obj, this will not block
            obj = self.socket.recv_pyobj(flags=zmq.NOBLOCK)
            # a message has been received
            return obj
        except zmq.Again as e:
            return None

training_complete = False
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = MNIST_NN(image_x, image_y, image_channel, output_channel).to(device)
batch_grad_norm = 0.0

ip = '127.0.0.1'
port = int(sys.argv[1])
timeout = 30
cpu = 60
data_amt = 100
connectivity = 70
client = client(ip, port, timeout, cpu, data_amt, connectivity)

client.start()

client.join()

print('Done.')