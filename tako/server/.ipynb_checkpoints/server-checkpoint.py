import zmq
import threading
import pandas as pd
import numpy as np
import time

'''
server
    Complete the FL workflow for 'round_num', and delete/join clients when completion
    1. Selection
        Select clients to participate training,
        clientMgr(server side) will send model to client(client side) automatically
    2. Aggregation
        After 'wait until completion', we can aggregate models in globally shared 'model_queue'
'''
class server(threading.Thread):
    def __init__(self, round_num, E, C, lock):
        threading.Thread.__init__(self)
        self.lock = lock
        self.client_E = E
        self.client_C = C
        self.round_num = round_num

    def run(self):
        global df, training, min_client, client_num, model
        # wait until enough client
        while(client_num < min_client):
            time.sleep(0.1)
        
        while(training):
            for round in range(round_num):
                # 'selection'
                # set their status to 'selected training'
                df.iloc[ df.sample(frac = C).index, df.columns.get_loc('status')] = 'selected training'
                print('selected training\n', df)
                
                # 'wait until completion'
                # status still having 'selected training' and 'training', we need to wait until complete
                while(len(df.loc[(df['status'] == 'training') | 
                                 (df['status'] == 'selected training')]) > 0):
                    time.sleep(0.1)
                
                print("client side complete!")
                
                # 'Aggregation'
                total_data_amt = 0
                for model_obj in model_queue:
                    data_amt = model_obj['data_amt']
                    model += data_amt * model_obj['model']
                    total_data_amt += data_amt
                
                model_queue.clear()
                model /= (total_data_amt + 1)
                print(f'Round {round} completed!')
            
            training = False
            
        '''
        join clients
        '''
        while(len(client_infos) > 0):
            client_info = client_infos.pop()
            client_info.join()
            del client_info