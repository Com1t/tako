import zmq
import pandas as pd

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
        training: client selected for training
        standby: client is able to join training
        lost: heartbeat lost, but not long enough
        dead: heartbeat expired for third of heartrate
    hash code:
        for identify
    client number:
        human readable number(monotonic increasing)
'''

df = pd.DataFrame(columns=["ip", "port", "sys_info", "status"])

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
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:7788")

p2p_connection_context = []
p2p_connection_socket = []
for i in range(2):
    df = df.append(socket.recv_pyobj())
    print(df)
    socket.send_pyobj('ack')
    
    # Estblish p2p Connection
    p2p_connection_context.append(zmq.Context())
    cur_context = p2p_connection_context[len(p2p_connection_context) - 1]
    
    p2p_connection_socket.append(cur_context.socket(zmq.PAIR))
    cur_socket = p2p_connection_socket[len(p2p_connection_socket) - 1]
    
    client_ip = df.iloc[len(df) - 1]['ip']
    client_port = df.iloc[len(df) - 1]['port']
    cur_socket.connect(f"tcp://{client_ip}:{client_port}")
    
    cur_socket.send_pyobj("connection built")
    print(cur_socket.recv_pyobj())

print("built p2p")
for i in range(10):
    cur_socket = p2p_connection_socket[i % 2]    
    cur_socket.send_pyobj("connection built")
    print(cur_socket.recv_pyobj())
    
# class connection(threading.Thread):
#     def __init__(self, lock):
#         threading.Thread.__init__(self)
#         self.lock = lock
#         self.connection_context = []
        
#     def run(self):
#         # 取得 lock
#         self.lock.acquire()
#         print("Lock acquired by Worker %d" % self.num)

#         # 不能讓多個執行緒同時進的工作
#         print("Worker %d: %s" % (self.num, msg))
#         time.sleep(1)

#         # 釋放 lock
#         print("Lock released by Worker %d" % self.num)
#         self.lock.release()
    
#     def registration():
        
        
#     def send():
        
        
#     def recv():
        
        
# class control_flow(threading.Thread):
#     def __init__(self, lock):
#         threading.Thread.__init__(self)
#         self.lock = lock

#     def run(self):
#         # 取得 lock
#         self.lock.acquire()
#         print("Lock acquired by Worker %d" % self.num)

#         # 不能讓多個執行緒同時進的工作
#         print("Worker %d: %s" % (self.num, msg))
#         time.sleep(1)

#         # 釋放 lock
#         print("Lock released by Worker %d" % self.num)
#         self.lock.release()