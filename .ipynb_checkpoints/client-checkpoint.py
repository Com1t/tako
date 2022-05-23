import zmq
import sys
import time
import json
import pandas as pd
import threading

class trainingThread(threading.Thread):
    def __init__(self, E):
        threading.Thread.__init__(self)
        self.E = E
    
    def run(self):
        global training_complete, model
        for epoch in range(self.E):
            time.sleep(2)
            model += 1
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
        global training_complete, model
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
                    model = msg['model']
                    # start training thread
                    training_complete = False
                    self.training_obj = trainingThread(epoch)
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
                           'model': model})
                
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
model = 0

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