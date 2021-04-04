import sys
import os
import glob
import csv
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from matplotlib import patches
from matplotlib.pyplot import figure
from datetime import timedelta, date
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score

#error occurs when directly import keras without tensorflow.python
from tensorflow.python.keras import layers, Input, regularizers, initializers
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping




class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)



def train_model(model, save_dir, task_id, train_x, train_y, valid_x, valid_y, test_x, test_y, batch_size=32, epochs=5):
    time_callback = TimeHistory()

    timelist = []
    # batchlist = [32,64,128,256,512,1024]
    batchlist = [32,64,128,256,512,1024,2048,4096,10000]

    for i in batchlist:
        print('starting batchsize {}'.format(i))

        history1 = model.fit(train_x, train_y, batch_size=i, epochs=2, validation_data=(valid_x, valid_y), verbose=2, callbacks=[time_callback])
        
        second_epoch = time_callback.times[-1]

        timelist = timelist + [second_epoch]

    print(timelist)

    return timelist, batchlist




task = pd.read_csv("task.csv") 


for index, row in task.iterrows():

    data_set = int(task['data_set'][index])

    save_dir = os.path.join(os.getcwd(), 'data_set/'+str(data_set))
    if not os.path.isdir(save_dir):
        #os.makedirs(save_dir)
        continue

    task_id = int(task['task_id'][index])
    input_size = int(task['input'][index])
    pred_k = int(task['k'][index])
    feature_num = int(task['feature_num'][index])
    label_threshold = float(task['label_threshold'][index])
    lstm_units = int(task['lstm_units'][index])
    lr = float(task['learning_rate'][index])
    regularizer = float(task['regularizer'][index])

    train_x = np.load(os.path.join(save_dir, 'train_x.npy'))
    train_y = np.load(os.path.join(save_dir, 'train_y_onehot.npy'))
    valid_x = np.load(os.path.join(save_dir, 'valid_x.npy'))
    valid_y = np.load(os.path.join(save_dir, 'valid_y_onehot.npy'))
    test_x = np.load(os.path.join(save_dir, 'test_x.npy'))
    test_y = np.load(os.path.join(save_dir, 'test_y_onehot.npy'))


    train_x = train_x[:10000]
    train_y = train_y[:10000]
    valid_x = valid_x[:10000]
    valid_y = valid_y[:10000]


    print('Running experiment {}'.format(task_id))


    #clear previous models
    clear_session()

    model_dir = os.path.join(os.getcwd(), 'load_model')

    if os.path.isdir(model_dir):
        model_dir = os.path.join(model_dir, str(task_id)+'/model/model_epoch_500.h5')
        if not os.path.isdir(model_dir):
            continue

        model = load_model(model_dir)
    else:

        #input_tensor = Input(shape=(30,4,1))
        input_tensor = Input(shape=(input_size,4,1))

        layer_x = layers.Conv2D(16, (1,4), kernel_regularizer=regularizers.l1(l=regularizer), kernel_initializer="zeros", bias_initializer="zeros")(input_tensor)
        layer_x = layers.BatchNormalization()(layer_x)
        layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

        layer_x = layers.Conv2D(16, (4,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer), kernel_initializer="zeros", bias_initializer="zeros")(layer_x)
        layer_x = layers.BatchNormalization()(layer_x)
        layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

        layer_x = layers.Conv2D(16, (4,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer), kernel_initializer="zeros", bias_initializer="zeros")(layer_x)
        layer_x = layers.BatchNormalization()(layer_x)
        layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)


        #dual input for ohlc+volume
        if feature_num == 5:
            train_x_ohlc = train_x[:,:,:4,:]
            train_x_volume = train_x[:,:,-1:,:]
            train_x = [train_x_ohlc, train_x_volume]

            valid_x_ohlc = valid_x[:,:,:4,:]
            valid_x_volume = valid_x[:,:,-1:,:]
            valid_x = [valid_x_ohlc, valid_x_volume]

            test_x_ohlc = test_x[:,:,:4,:]
            test_x_volume = test_x[:,:,-1:,:]
            test_x = [test_x_ohlc, test_x_volume]


            input_tensor2 = Input(shape=(input_size,1,1))

            layer_x2 = layers.Conv2D(16, (1,1), kernel_regularizer=regularizers.l1(l=regularizer), kernel_initializer="zeros", bias_initializer="zeros")(input_tensor2)
            layer_x2 = layers.BatchNormalization()(layer_x2)
            layer_x2 = layers.LeakyReLU(alpha=0.01)(layer_x2)

            layer_x2 = layers.Conv2D(16, (4,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer), kernel_initializer="zeros", bias_initializer="zeros")(layer_x2)
            layer_x2 = layers.BatchNormalization()(layer_x2)
            layer_x2 = layers.LeakyReLU(alpha=0.01)(layer_x2)

            layer_x2 = layers.Conv2D(16, (4,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer), kernel_initializer="zeros", bias_initializer="zeros")(layer_x2)
            layer_x2 = layers.BatchNormalization()(layer_x2)
            layer_x2 = layers.LeakyReLU(alpha=0.01)(layer_x2)

            layer_x = layers.concatenate([layer_x, layer_x2], axis=-1)



        # Inception Module
        tower_1 = layers.Conv2D(32, (1,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer), kernel_initializer="zeros", bias_initializer="zeros")(layer_x)
        tower_1 = layers.BatchNormalization()(tower_1)
        tower_1 = layers.LeakyReLU(alpha=0.01)(tower_1)
        tower_1 = layers.Conv2D(32, (3,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer), kernel_initializer="zeros", bias_initializer="zeros")(tower_1)
        tower_1 = layers.BatchNormalization()(tower_1)
        tower_1 = layers.LeakyReLU(alpha=0.01)(tower_1)

        tower_2 = layers.Conv2D(32, (1,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer), kernel_initializer="zeros", bias_initializer="zeros")(layer_x)
        tower_2 = layers.BatchNormalization()(tower_2)
        tower_2 = layers.LeakyReLU(alpha=0.01)(tower_2)
        tower_2 = layers.Conv2D(32, (5,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer), kernel_initializer="zeros", bias_initializer="zeros")(tower_2)
        tower_2 = layers.BatchNormalization()(tower_2)
        tower_2 = layers.LeakyReLU(alpha=0.01)(tower_2)  

        tower_3 = layers.MaxPooling2D((3,1), padding='same', strides=(1,1))(layer_x)
        tower_3 = layers.Conv2D(32, (1,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer), kernel_initializer="zeros", bias_initializer="zeros")(tower_3)
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
            layer_x = layers.CuDNNLSTM(lstm_units, kernel_regularizer=regularizers.l1(l=regularizer), kernel_initializer="zeros",recurrent_initializer="zeros",bias_initializer="zeros")(layer_x)
        # if using CPU
        else:
            print('using CPU')
            layer_x = layers.LSTM(lstm_units, kernel_regularizer=regularizers.l1(l=regularizer), kernel_initializer="zeros",recurrent_initializer="zeros",bias_initializer="zeros")(layer_x)

        # # The last output layer uses a softmax activation function
        output = layers.Dense(3, activation='softmax', kernel_initializer="zeros", bias_initializer="zeros")(layer_x)


        if feature_num == 4:
            model = Model(input_tensor, output)

        elif feature_num == 5:
            model = Model([input_tensor, input_tensor2], output)


        opt = Adam(lr=lr, epsilon=1)# learning rate and epsilon are the same as paper DeepLOB 
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 
        #model.summary()



    save_dir = os.path.join(os.getcwd(), 'result/'+str(task_id))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)        


    timelist, batchlist = train_model(model, save_dir, task_id, train_x, train_y, valid_x, valid_y, test_x, test_y, batch_size=1, epochs=1)

    if lstm_units == 1:
        time_1 = timelist
    elif lstm_units == 16:
        time_16 = timelist
    elif lstm_units == 32:
        time_32 = timelist
    elif lstm_units == 64:
        time_64 = timelist



plt.xticks(batchlist)
plt.plot(time_1)
plt.plot(time_16)
plt.plot(time_32)
plt.plot(time_64)
plt.title('time_gpu')
plt.ylabel('Time(sec)')
plt.xlabel('Batch size')
plt.legend(['lstm=1', 'lstm=16', 'lstm=32', 'lstm=64'])
# plt.savefig('time_gpu_lstm_allrecord.png')
# plt.savefig('time_cpu_10000_fullepoch.png')
plt.savefig('time_cpu_10000_fullepoch_new.png')
plt.clf()

