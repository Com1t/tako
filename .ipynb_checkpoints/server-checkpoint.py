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
min_client = 2
training = True
timeout = np.timedelta64(1,'m')

# client info
client_infos = []

# models
model = 10
model_queue = []
model_lock = threading.Lock()

# client training related
round_num = 2
C = 0.5
E = 3


'''
Client
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

class client(threading.Thread):
    def __init__(self, context, socket, num, lock):
        global df
        threading.Thread.__init__(self)
        self.context = context
        self.socket = socket
        self.num = num
        self.lock = lock
        df.iloc[self.num, df.columns.get_loc('heartbeat')] = np.datetime64('now')
        
    def run(self):
        global df, model, training
        while(training):
            time.sleep(0.1)
            # check heartbeat
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
                print(f'{self.num} selected training {model}')
                self.send({'E': E, 'model': model})
                df.iloc[self.num, df.columns.get_loc('status')] = 'training'
                
            elif status == 'training':
                if(msg != False and msg != 'heartbeat'):
                    # it must be model
                    print(f'{self.num} received {msg} {type(msg)}')
                    for info in msg['sys_info'].keys():
                        df.iloc[self.num, df.columns.get_loc(info)] = msg['sys_info'][info]
                    
                    # since this queue is globaly shared, we have to ensure concurrency
                    self.lock.acquire()
                    model_queue.append({'data_amt': msg['sys_info']['data amount'], 'model': msg['model']})
                    print(model_queue)
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
            return False

'''
Estblish Connection
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
        
        # server variable
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:7788")
        
    def run(self):
        # create new context or not        
        global df, client_num, training, client_infos
        while(training):
            recv_df = self.recv()
            if recv_df is not None:
                print('new client info', recv_df)
                self.send('ack')

                # Estblish p2p Connection
                cur_context = zmq.Context()
                cur_socket = cur_context.socket(zmq.PAIR)
                client_ip = recv_df.iloc[0]['ip']
                client_port = recv_df.iloc[0]['port']
                cur_socket.connect(f"tcp://{client_ip}:{client_port}")

                cur_socket.send_pyobj("server side connection built")

                # if success, we will receive an ack msg
                if "client side connection built" == cur_socket.recv_pyobj():
                    df = df.append(recv_df, ignore_index=True)
                    df.iloc[client_num,df.columns.get_loc('status')] = 'standby'
                    client_infos.append(client(cur_context, cur_socket, client_num, self.lock))
                    client_infos[client_num].start()
                    time.sleep(0.2)
                    print('New client registered')
                    client_num += 1;
                # if not we revert
                else:
                    print("Connection Error")
        print('Connection complete')
        
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
        

class server(threading.Thread):
    def __init__(self, round_num, E, C, lock):
        threading.Thread.__init__(self)
        self.lock = lock
        self.client_E = E
        self.client_C = C
        self.round_num = round_num

    def run(self):
        global df, training, min_client, model
        # wait until enough client
        while(client_num < min_client):
            time.sleep(0.1)
        
#         while(training):
#             inst = input('getting inst ')
#             if(inst == 'send'):
#                 dest = input('getting dest ')
#                 model = input('getting model ')
#                 df.iloc[int(dest),df.columns.get_loc('status')] = 'selected training'
#             if(inst == 'show'):
#                 print(df)
        
        while(training):
            for round in range(round_num):
                # select client, and set their status to 'selected training'
                df.iloc[ df.sample(frac = C).index, df.columns.get_loc('status')] = 'selected training'
                print('selected training', df)
                
                # status still having 'selected training' and 'training', we need to wait until complete
                while(len(df.loc[(df['status'] == 'training') | 
                                 (df['status'] == 'selected training')]) > 0):
                    time.sleep(0.1)
                
#                 print(df.loc[(df['status'] == 'training') | 
#                              (df['status'] == 'selected training')])
                
                print("complete client side")
                
                # aggregation
                total_data_amt = 0
                print(len(model_queue))
                for model_obj in model_queue:
                    data_amt = model_obj['data_amt']
                    model += data_amt * model_obj['model']
                    total_data_amt += data_amt
                
                model_queue.clear()
                model /= (total_data_amt + 1)
                print("complete")
            
            training = False
            
        '''
        join clients
        '''
        while(len(client_infos) > 0):
            client_info = client_infos.pop()
            client_info.join()
            del client_info
        print('join complete')

connectionMgr = connectionMgr(model_lock)
server = server(round_num, E, C, model_lock)

connectionMgr.start()
server.start()


server.join()
print('server joined')
connectionMgr.join()
print('connectionMgr joined')

print("Done.")