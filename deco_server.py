import zmq
import threading
import pandas as pd
import numpy as np
import tako.server as server 

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
round_num = 20
C = 1
E = 3

connectionMgr = server.connectionMgr(model_lock)
server = server.server(round_num, E, C, model_lock)

connectionMgr.start()
server.start()


server.join()
connectionMgr.join()

print("Done.")