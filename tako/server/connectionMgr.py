import zmq
import threading
import pandas as pd
import numpy as np
import time

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