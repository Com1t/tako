import zmq
import sys
import time
import pandas as pd

port = int(sys.argv[1])
df = pd.DataFrame({
    "ip": "127.0.0.1",
    "port": port,
    "cpu": 30, 
    "connectivity": 60
}, index=[0])

# Estblish Connection
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect ("tcp://127.0.0.1:7788")
socket.send_pyobj(df)
print(socket.recv_pyobj())

# close if server acked
socket.close()

socket = context.socket(zmq.PAIR)
socket.bind(f"tcp://*:{port}")
print(socket.recv_pyobj())
socket.send_pyobj("client side connection built")

def recv():
    global socket
    try:
        #check for a obj, this will not block
        obj = socket.recv_pyobj(flags=zmq.NOBLOCK)
        # a message has been received
        return obj
    except zmq.Again as e:
        return False

heartrate = 0
while(True):
    msg = recv()
    if(msg != False):
        print(msg)
        socket.send_pyobj("ack")
    time.sleep(0.2)
    heartrate += 1
    if(heartrate == 30):
        heartrate = 0
        socket.send_pyobj("heartbeat")
        print("heartbeat")