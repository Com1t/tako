import zmq
import threading
import pandas as pd
import numpy as np
import time

'''
Server should be a well known port
Client connects to server, server does the registration

Server use MQTT to subscribe clients heartbeat to keep on track

While one thread accounting connections,
Server will spawn sub-thread to deal with aggregation
Client will spawn sub-thread to train model

Client Registration:
    Client will send us
    IP address
    port
    System information (currently empty)

While server will keep more attribute
    status:
        selected training: selected not yet send model
        training: client selected for training
        standby: client is able to join training
        lost: heartbeat lost, but not long enough
    client number:
        human readable number(monotonic increasing)(index will do the job)
'''

# client database
dtypes = np.dtype(
    [
        ("ip", str),
        ("port", int),
        ("status", str),
        ("heartbeat", np.datetime64),
    ]
)
df = pd.DataFrame(np.empty(0, dtype=dtypes))

client_num = 0
min_client = 3
training = True
timeout = np.timedelta64(1,'m')

# client info
client_infos = []

# models
model = 10
model_queue = []
model_lock = threading.Lock()

# client training related
round_num = 20
C = 0.5
E = 1


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
    
'''
clientMgr
    share the global dataframe, it will check it's own state to get information it needs
    Generally, two thing will be concluded constantly
    1. Check heart beat
        It will receive the heartbeat signal, if it didn't receive any for longer than timeout
        The client state will become 'lost'
    2. Check its own status, do something corresponding
        Currently, two type of status can trigger event
        a. 'selected training'
            If selected, it will send current global model to client,
            and jump to 'training' status
        b. 'training'
            A client in training means it sent the global model to client.
            Currently waiting client training complete
            At the moment of completion, it will put the received model into 'model_queue',
            and update some 'sys_info', in the globaly shared dataframe
'''

class clientMgr(threading.Thread):
    def __init__(self, context, socket, num, lock):
        global df
        threading.Thread.__init__(self)
        # globaly shared lock
        self.lock = lock
        
        self.context = context
        self.socket = socket
        self.num = num
        df.iloc[self.num, df.columns.get_loc('heartbeat')] = np.datetime64('now')
        
    def run(self):
        global df, model, training
        while(training):
            time.sleep(0.1)
            # check receive message
            msg = self.recv()
            
            # check heartbeat
            if(msg == 'heartbeat'):
                print(f"heartbeat {self.num}")
                past = df.iloc[self.num, df.columns.get_loc('heartbeat')]
                now = np.datetime64('now')
                if(now - past > timeout):
                    df.iloc[self.num, df.columns.get_loc('status')] = 'lost'
                else:
                    df.iloc[self.num, df.columns.get_loc('heartbeat')] = now
            
            # check FL workflow
            status = df.iloc[self.num]['status']
            if status == 'selected training':
                print(f'client {self.num} selected training')
                self.send({'E': E, 'model': model.state_dict()})
                df.iloc[self.num, df.columns.get_loc('status')] = 'training'
                
            elif status == 'training':
                if(msg is not None and msg != 'heartbeat'):
                    # it must be model
                    print(f'client {self.num} received')
                    for info in msg['sys_info'].keys():
                        df.iloc[self.num, df.columns.get_loc(info)] = msg['sys_info'][info]
                    
                    # since this queue is globaly shared, we have to ensure concurrency
                    self.lock.acquire()
                    model_queue.append({'data_amt': msg['sys_info']['data amount'], 
                                        'model': msg['model']})
                    self.lock.release()
                    
                    df.iloc[self.num, df.columns.get_loc('status')] = 'standby'
                
        self.send("training complete")
        
    def __del__(self):
        self.socket.close()
        self.context.term()
     
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

'''
connectionMgr
    Registrate new cleint, and build p2p connection with client
        Two key component on server
        1. Initial Server Connection
            Every client can reach server by this connection,
            while this connection is only for establishing a p2p connection
        2. p2p connection
            After built initial server connection, server will build an p2p connection
            This can provide a clear way to communicate between server and client
'''
class connectionMgr(threading.Thread):
    def __init__(self, lock):
        threading.Thread.__init__(self)
        # globaly shared lock
        self.lock = lock
        
        # connectionMgr server variable
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:7788")
        
    def run(self):       
        global df, client_num, training, client_infos
        while(training):
            # wait a bit
            time.sleep(0.1)
            recv_df = self.recv()
            if recv_df is not None:
                # in server side the only message that flows to here will be registration msg
                print('new client info\n', recv_df)
                self.send('ack')

                # Estblish p2p Connection
                cur_context = zmq.Context()
                cur_socket = cur_context.socket(zmq.PAIR)
                client_ip, client_port = recv_df.iloc[0]['ip'], recv_df.iloc[0]['port']
                cur_socket.connect(f"tcp://{client_ip}:{client_port}")

                cur_socket.send_pyobj("server side connection built")

                # if success, we will receive an ack msg
                if "client side connection built" == cur_socket.recv_pyobj():
                    df = df.append(recv_df, ignore_index=True)
                    df.iloc[client_num,df.columns.get_loc('status')] = 'standby'
                    client_infos.append(clientMgr(cur_context, cur_socket, client_num, self.lock))
                    client_infos[client_num].start()
                    time.sleep(0.2)
                    print('New client registered')
                    client_num += 1;
                # if not we revert
                else:
                    print("Connection Error")
        
    def __del__(self):
        self.socket.close()
        self.context.term()
     
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
        
'''
server
    Complete the FL workflow for 'round_num', and delete/join clients when completion
    1. Selection
        Select clients to participate training,
        clientMgr(server side) will send model to client(client side) automatically
    2. Aggregation
        After 'wait until completion', we can aggregate models in globally shared 'model_queue'
'''
class server(threading.Thread):
    def __init__(self, round_num, E, C, lock):
        threading.Thread.__init__(self)
        self.lock = lock
        self.client_E = E
        self.client_C = C
        self.round_num = round_num

    def run(self):
        global df, training, min_client, model, device
        # wait until enough client
        while(client_num < min_client):
            time.sleep(0.1)
            
        from torch_npz.FLDataset import FLDataset
        test_data = FLDataset('/ML/FL_algo/nonIIDdataset/client_1.pickle', 'test')
        testLoader = dset.DataLoader(test_data, batch_size=1024, shuffle=False)
        
        while(training):
            for round in range(round_num):
                # save model
                # torch.save(model.state_dict(), PATH)
                # 'selection'
                # set their status to 'selected training'
                df.iloc[ df.sample(frac = C).index, df.columns.get_loc('status')] = 'selected training'
                print('selected training\n', df)
                
                # 'wait until completion'
                # status still having 'selected training' and 'training', we need to wait until complete
                while(len(df.loc[(df['status'] == 'training') | 
                                 (df['status'] == 'selected training')]) > 0):
                    time.sleep(0.1)
                
                print("client side complete!")
                
                # 'Aggregation'
                total_data_amt = 0
                for model_obj in model_queue:
                    data_amt = model_obj['data_amt']
                    
                    model_dict = model.state_dict()
                    for k in model_dict.keys():
                        model_dict[k] = torch.stack([model_queue[i]['model'][k].float()
                                                      for i in range(len(model_queue))], 0).mean(0)
                    model.load_state_dict(model_dict)
                    total_data_amt += data_amt
                
                model_queue.clear()
                with torch.no_grad():
                    correct_count = 0
                    for _, (data) in enumerate(testLoader):
                        image = data[0].to(device)
                        labels = data[1].to(device)
                        output = model(image)
                        predict_label = torch.argmax(nn.Softmax(dim=1)(output), dim=1, keepdim=False)
                        correct_count += (predict_label == labels).float().sum()
                    acc = correct_count / len(test_data)
                    print('Acc : {:.4f}'.format(acc.item()))
                print(f'Round {round} completed!')
            
            training = False
            
        '''
        join clients
        '''
        while(len(client_infos) > 0):
            client_info = client_infos.pop()
            client_info.join()
            del client_info

device = 'cpu'
model = MNIST_NN(image_x, image_y, image_channel, output_channel).to(device)

connectionMgr = connectionMgr(model_lock)
server = server(round_num, E, C, model_lock)

connectionMgr.start()
server.start()


server.join()
connectionMgr.join()

print("Done.")