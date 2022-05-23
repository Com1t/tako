import zmq
import threading
import pandas as pd
import numpy as np
import time

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
                print(f'client {self.num} selected training {model}')
                self.send({'E': E, 'model': model})
                df.iloc[self.num, df.columns.get_loc('status')] = 'training'
                
            elif status == 'training':
                if(msg is not None and msg != 'heartbeat'):
                    # it must be model
                    print(f'client {self.num} received {msg} {type(msg)}')
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
