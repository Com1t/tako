import zmq
import pandas as pd

'''
Server should be a well known port
Client connects to server, server does the registration

Server use MQTT to subscribe clients heartbeat to keep on track

While one thread accounting connections,
Server will spawn sub-thread to deal with aggregation
Client will spawn sub-thread to train model
'''

# Client Registration
# Client will send us
# IP address, port
# System information (currently empty)
df = pd.DataFrame(columns=["ip", "port", "sys_info", "status"])

# While server will keep more attribute
# such as
# status:
# training: client selected for training
# standby: client is able to join training
# lost: heartbeat lost, but not long enough
# dead: heartbeat expired for third of heartrate
# hash code:
# for identify
# client number:
# human readable number(monotonic increasing)

# Estblish Connection
for i in range(2):
    context = []
    context.append(zmq.Context())
    cur_context = context[len(context) - 1]
    socket = cur_context.socket(zmq.REP)
    socket.bind("tcp://*:7788")

    df = df.append(socket.recv_pyobj())

    print(df)
    socket.send_pyobj('ack')

print(df)
# wait for request(other thread)
# print(socket.recv_pyobj())

# initialize model
# class connection(threading.Thread):
#     def __init__(self, queue, num, lock):
#         threading.Thread.__init__(self)
#         self.num = num
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
        
# class control_flow(threading.Thread):
#     def __init__(self, queue, num, lock):
#         threading.Thread.__init__(self)
#         self.num = num
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