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

print(type(x_train), type(x_test))
print(x_train.shape, x_test.shape)
