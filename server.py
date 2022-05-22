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

# shared variables
df = pd.DataFrame(columns=["ip", "port", "sys_info", "status", "heartbeat"])

dtypes = np.dtype(
    [
        ("ip", str),
        ("port", int),
        ("sys_info", float),
        ("status", str),
        ("heartbeat", np.datetime64),
    ]
)
df = pd.DataFrame(np.empty(0, dtype=dtypes))

client_num = 0
min_client = 1
training = True
timeout = np.timedelta64(1,'m')

# client info
client_infos = []

model = 'model'

'''
Client
    share the global dataframe, it will check it's own state to get information it needs
    async recv?
'''

class client(threading.Thread):
    def __init__(self, context, socket, num):
        threading.Thread.__init__(self)
        self.context = context
        self.socket = socket
        self.num = num
        df.iloc[self.num, df.columns.get_loc('heartbeat')] = np.datetime64('now')
        
    def run(self):
        global model, training
        while(training):
            time.sleep(0.1)
            # check heartbeat
            msg = self.recv()
            
            if(msg == 'heartbeat'):
                print(f"heartbeat {self.num}")
                past = df.iloc[self.num, df.columns.get_loc('heartbeat')]
                now = np.datetime64('now')
                if(now - past > timeout):
                    df.iloc[self.num, df.columns.get_loc('status')] = 'lost'
                else:
                    df.iloc[self.num, df.columns.get_loc('heartbeat')] = now
                
            status = df.iloc[self.num]['status']
            if status == 'selected training':
                print(f'{self.num} selected training {model}')
                self.send(model)
                df.iloc[self.num, df.columns.get_loc('status')] = 'training'
                
            elif status == 'training':
                if(msg != False and msg != 'heartbeat'):
                    print(f'{self.num} received {msg}')
                    df.iloc[self.num, df.columns.get_loc('status')] = 'standby'
                
        send("training complete")
        
    def __del__(self):
        self.socket.close()
        self.context.term()
 
    def send(self, obj):
        self.socket.send_pyobj(obj)
        
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
        # share variable
        self.lock = lock
        
        # server variable
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:7788")
        
    def run(self):
        # create new context or not        
        global df, client_num, training, client_infos
        while(training):
            recv_df = self.socket.recv_pyobj()
            print(recv_df)
            self.socket.send_pyobj('ack')

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
                client_infos.append(client(cur_context, cur_socket, client_num))
                client_infos[client_num].start()
                time.sleep(0.2)
                print(df)
                client_num += 1;
            # if not we revert
            else:
                print("Connection Error")
        

class server(threading.Thread):
    def __init__(self, lock):
        threading.Thread.__init__(self)
        self.lock = lock

    def run(self):
        global df, training, min_client, model
        # wait until enough client
        while(client_num < min_client):
            time.sleep(0.1)
        print("able")
        while(training):
            inst = input('getting inst ')
            if(inst == 'send'):
                dest = input('getting dest ')
                model = input('getting model ')
                df.iloc[int(dest),df.columns.get_loc('status')] = 'selected training'
            if(inst == 'show'):
                print(df)
                
#         while(training)
#             '''
#                 find status 'new_born' and initialized them
#             '''
            
#             while
#                 do training
#                 select client
#                 while
#                     wait model 
#                 aggregation
        
#         '''
#         join clients
#         '''

lock = threading.Lock()
connectionMgr = connectionMgr(lock)
server = server(lock)

connectionMgr.start()
server.start()

server.join()
connectionMgr.join()

print("Done.")