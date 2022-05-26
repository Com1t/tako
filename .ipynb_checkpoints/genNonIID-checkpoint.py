# %%
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import random
import os
import pickle

client_num = 28
shards_per_client = 2
image_per_shard = int(60000 / (client_num * shards_per_client))

dataset_type = 'IID'

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Sorting dataset
Sorted = []
print(f'generating {dataset_type} data')
if dataset_type == 'IID':
    for i in range(len(x_train)):
        Sorted.append((x_train[i], y_train[i]))
    
elif dataset_type == 'non IID': 
    toSort = {}
    for i in range(10):
        toSort[i] = []

    for x, y in zip(x_train, y_train):
        toSort[y].append(x)

    # Count of Each number
    for i in range(10):
        print(f'label {i}: {len(toSort[i])}')

    # Get the into one single array
    for i in range(10):
        for j in toSort[i]:
            Sorted.append((j, i))
            
print(f'len(Sorted): {len(Sorted)}')

# shards
shards = []
for i in range(client_num * shards_per_client):
    data = Sorted[i * image_per_shard:(i + 1) * image_per_shard]
    shards.append(data)
print(f'len(shards): {len(shards)}')

# Shuffle shards
random.shuffle(shards)

# Arrange to each client
os.mkdir('dataset')
for i in range(client_num):
    data = []
    for j in range(shards_per_client):
        data += shards[i * shards_per_client + j]
        xs = np.array([sample[0] for sample in data])
        ys = np.array([sample[1] for sample in data])
        
    with open('dataset/client_%d.pickle' % (i + 1), 'wb') as fp:
        pickle.dump((xs, ys, x_test, y_test), fp)
    
exit()