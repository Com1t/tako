import zmq
import pandas as pd

df = pd.DataFrame({
    "ip": "127.0.0.1",
    "port": 38010,
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
socket.bind("tcp://*:38010")
print(socket.recv_pyobj())
socket.send_pyobj("client connection built ack")

for i in range(5):
    print(socket.recv_pyobj())
    socket.send_pyobj("client connection built ack")