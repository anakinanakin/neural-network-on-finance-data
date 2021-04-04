import psycopg2, psycopg2.extras
import os
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from matplotlib import patches
from matplotlib.pyplot import figure
from datetime import timedelta, date
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score

#error occurs when directly import keras without tensorflow.python
from tensorflow.python.keras import layers, Input, regularizers
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping



task_id = 1
input_size = 30
pred_k = 30


lstm_units = 32
lr = 0.01
regularizer = 0


save_dir = os.path.join(os.getcwd(), 'data_set/'+str(7))

train_x = np.load(os.path.join(save_dir, 'test_x.npy'))
train_y = np.load(os.path.join(save_dir, 'test_y_onehot.npy'))
valid_x = np.load(os.path.join(save_dir, 'valid_x.npy'))
valid_y = np.load(os.path.join(save_dir, 'valid_y_onehot.npy'))
test_x = np.load(os.path.join(save_dir, 'valid_x.npy'))
test_y = np.load(os.path.join(save_dir, 'valid_y_onehot.npy'))



#input_tensor = Input(shape=(30,4,1))
input_tensor = Input(shape=(input_size,4,1))

layer_x = layers.Conv2D(16, (1,4), kernel_regularizer=regularizers.l1(l=regularizer))(input_tensor)
layer_x = layers.BatchNormalization()(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

layer_x = layers.Conv2D(16, (4,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)
layer_x = layers.BatchNormalization()(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

layer_x = layers.Conv2D(16, (4,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)
layer_x = layers.BatchNormalization()(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)




# Inception Module
tower_1 = layers.Conv2D(32, (1,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)
tower_1 = layers.BatchNormalization()(tower_1)
tower_1 = layers.LeakyReLU(alpha=0.01)(tower_1)
tower_1 = layers.Conv2D(32, (3,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(tower_1)
tower_1 = layers.BatchNormalization()(tower_1)
tower_1 = layers.LeakyReLU(alpha=0.01)(tower_1)

tower_2 = layers.Conv2D(32, (1,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)
tower_2 = layers.BatchNormalization()(tower_2)
tower_2 = layers.LeakyReLU(alpha=0.01)(tower_2)
tower_2 = layers.Conv2D(32, (5,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(tower_2)
tower_2 = layers.BatchNormalization()(tower_2)
tower_2 = layers.LeakyReLU(alpha=0.01)(tower_2)  

tower_3 = layers.MaxPooling2D((3,1), padding='same', strides=(1,1))(layer_x)
tower_3 = layers.Conv2D(32, (1,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(tower_3)
tower_3 = layers.BatchNormalization()(tower_3)
tower_3 = layers.LeakyReLU(alpha=0.01)(tower_3)

layer_x = layers.concatenate([tower_1, tower_2, tower_3], axis=-1)

# concatenate features of tower_1, tower_2, tower_3
layer_x = layers.Reshape((input_size,96))(layer_x)
#layer_x = layers.Reshape((input_size,feature_num))(input_tensor)

# # 64 LSTM units
#layer_x = layers.LSTM(64)(layer_x)

# if using GPU
if tf.test.is_gpu_available():
	print('using GPU')
	layer_x = layers.CuDNNLSTM(lstm_units, kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)
# if using CPU
else:
	print('using CPU')
	layer_x = layers.LSTM(lstm_units, kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)

# # The last output layer uses a softmax activation function
output = layers.Dense(3, activation='softmax')(layer_x)



model = Model(input_tensor, output)


opt = Adam(lr=lr, epsilon=1)# learning rate and epsilon are the same as paper DeepLOB 
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 
model.summary()
