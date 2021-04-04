import sys
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score, log_loss

#error occurs when directly import keras without tensorflow.python
from tensorflow.python.keras import layers, Input, regularizers
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping


#change all is_gpu_available() to this global variable
#pass sys.argv in main()
device_type = 'gpu'
# device_type = 'cpu'


def get_f1_pre_recall(model, x, y, batch_size):
    y_pred = model.predict(x, verbose=2, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred.tolist()

    f1 = f1_score(y, y_pred, average='macro')
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')

    return f1, precision, recall


def train_model(model, save_dir, task_id, train_x, train_y, valid_x, valid_y, batch_size=512, epochs=500, checkpoint=50):

    save_dir_model = os.path.join(save_dir, 'model')
    if not os.path.isdir(save_dir_model):
        os.makedirs(save_dir_model)

    save_dir_output = os.path.join(save_dir, 'output')
    if not os.path.isdir(save_dir_output):
        os.makedirs(save_dir_output)

    train_acc = []
    valid_acc = []

    train_loss = []
    valid_loss = []

    train_f1 = []
    valid_f1 = []

    train_precision = []
    valid_precision = []

    train_recall = []
    valid_recall = []

    train_y_label = [np.where(r==1)[0][0] for r in train_y]
    valid_y_label = [np.where(r==1)[0][0] for r in valid_y]

    for i in range(epochs):
        print('starting epoch {}'.format(i+1))
        history1 = model.fit(train_x, train_y, batch_size=batch_size, epochs=1, validation_data=(valid_x, valid_y), verbose=2)

        if (i+1)%checkpoint==0:
            model_dir = 'result/{}/'.format(task_id)+'model/model_epoch_{}.h5'.format(i+1)
            #model.save(os.path.join(save_dir,'model/model_epoch:{}.h5'.format(i+1)))
            if os.path.isfile(model_dir):
                os.remove(model_dir)
            model.save(model_dir)

        train_acc =  train_acc  + history1.history['acc']
        train_loss = train_loss + history1.history['loss']
        valid_acc =  valid_acc  + history1.history['val_acc']
        valid_loss = valid_loss + history1.history['val_loss']

        f1, precision, recall = get_f1_pre_recall(model, train_x, train_y_label, batch_size)

        train_f1 = train_f1 + [f1]
        train_precision = train_precision + [precision]
        train_recall = train_recall + [recall]

        f1, precision, recall = get_f1_pre_recall(model, valid_x, valid_y_label, batch_size)

        valid_f1 = valid_f1 + [f1]
        valid_precision = valid_precision + [precision]
        valid_recall = valid_recall + [recall]

        if (i+1)%checkpoint==0:
            train_acc_np = np.array(train_acc)
            valid_acc_np = np.array(valid_acc)

            train_loss_np = np.array(train_loss)
            valid_loss_np = np.array(valid_loss)

            train_f1_np = np.array(train_f1)
            valid_f1_np = np.array(valid_f1)

            train_precision_np = np.array(train_precision)
            valid_precision_np = np.array(valid_precision)

            train_recall_np = np.array(train_recall)
            valid_recall_np = np.array(valid_recall)

            np.save(os.path.join(save_dir, 'output/train_acc.npy'), train_acc_np)
            np.save(os.path.join(save_dir, 'output/valid_acc.npy'), valid_acc_np)

            np.save(os.path.join(save_dir, 'output/train_loss.npy'), train_loss_np)
            np.save(os.path.join(save_dir, 'output/valid_loss.npy'), valid_loss_np)

            np.save(os.path.join(save_dir, 'output/train_f1.npy'), train_f1_np)
            np.save(os.path.join(save_dir, 'output/valid_f1.npy'), valid_f1_np)

            np.save(os.path.join(save_dir, 'output/train_precision.npy'), train_precision_np)
            np.save(os.path.join(save_dir, 'output/valid_precision.npy'), valid_precision_np)

            np.save(os.path.join(save_dir, 'output/train_recall.npy'), train_recall_np)
            np.save(os.path.join(save_dir, 'output/valid_recall.npy'), valid_recall_np)


def main(thread_num):
    # causing allocating 2 gpu devices 
    # if tf.test.is_gpu_available():

    # select gpu device 0,1
    if device_type == 'gpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = thread_num


    task = pd.read_csv('task.csv') 


    for index, row in task.iterrows():

        # multi-gpu 
        # if device_type == 'gpu':
        #     if index%2 != int(thread_num):
        #         continue

        completed = int(task['completed'][index])
        if completed == 1:
            continue

        data_set = int(task['data_set'][index])

        load_dir = os.path.join(os.getcwd(), 'data_set/'+str(data_set))
        if not os.path.isdir(load_dir):
            continue

        task_id = int(task['task_id'][index])
        input_size = int(task['input'][index])
        pred_k = int(task['k'][index])
        feature_num = int(task['feature_num'][index])
        label_threshold = float(task['label_threshold'][index])
        lstm_units = int(task['lstm_units'][index])
        lr = float(task['learning_rate'][index])
        epsilon = float(task['epsilon'][index])
        regularizer = float(task['regularizer'][index])

        train_x = np.load(os.path.join(load_dir, 'train_x.npy'))
        train_y = np.load(os.path.join(load_dir, 'train_y_onehot.npy'))
        valid_x = np.load(os.path.join(load_dir, 'valid_x.npy'))
        valid_y = np.load(os.path.join(load_dir, 'valid_y_onehot.npy'))

        print("Running experiment {}".format(task_id))

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

            layer_x = layers.Conv2D(16, (1,4), kernel_regularizer=regularizers.l1(l=regularizer))(input_tensor)
            layer_x = layers.BatchNormalization()(layer_x)
            layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

            layer_x = layers.Conv2D(16, (4,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)
            layer_x = layers.BatchNormalization()(layer_x)
            layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

            layer_x = layers.Conv2D(16, (4,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)
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


                input_tensor2 = Input(shape=(input_size,1,1))

                layer_x2 = layers.Conv2D(16, (1,1), kernel_regularizer=regularizers.l1(l=regularizer))(input_tensor2)
                layer_x2 = layers.BatchNormalization()(layer_x2)
                layer_x2 = layers.LeakyReLU(alpha=0.01)(layer_x2)

                layer_x2 = layers.Conv2D(16, (4,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(layer_x2)
                layer_x2 = layers.BatchNormalization()(layer_x2)
                layer_x2 = layers.LeakyReLU(alpha=0.01)(layer_x2)

                layer_x2 = layers.Conv2D(16, (4,1), padding='same', kernel_regularizer=regularizers.l1(l=regularizer))(layer_x2)
                layer_x2 = layers.BatchNormalization()(layer_x2)
                layer_x2 = layers.LeakyReLU(alpha=0.01)(layer_x2)

                layer_x = layers.concatenate([layer_x, layer_x2], axis=-1)



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
            if device_type == 'gpu':
                print('using GPU')
                layer_x = layers.CuDNNLSTM(lstm_units, kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)
            # if using CPU
            elif device_type == 'cpu':
                print('using CPU')
                layer_x = layers.LSTM(lstm_units, kernel_regularizer=regularizers.l1(l=regularizer))(layer_x)
            else:
                sys.exit("wrong device type")

            # # The last output layer uses a softmax activation function
            output = layers.Dense(2, activation='softmax')(layer_x)


            if feature_num == 4:
                model = Model(input_tensor, output)

            elif feature_num == 5:
                model = Model([input_tensor, input_tensor2], output)


            opt = Adam(lr=lr, epsilon=epsilon)
            model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 
            #model.summary()


        save_dir = os.path.join(os.getcwd(), 'result/'+str(task_id))
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)        

        train_model(  model, \
                      save_dir, \
                      task_id, \
                      train_x, \
                      train_y, \
                      valid_x, \
                      valid_y, \
                      batch_size=512, \
                      epochs=500, \
                      checkpoint=100)


        with open(os.path.join(save_dir, 'readme.txt'), 'w') as f:
            f.write("""'task id = {}\n
                        input size = {}\n
                        prediction k = {}\n
                        feature = {}\n
                        label threshold = {}\n
                        lstm units = {}\n
                        learning rate = {}\n
                        epsilon = {}\n
                        regularizer = {}\n
                        data set = {}'""".format(task_id, \
                                                input_size, \
                                                pred_k, \
                                                feature_num, \
                                                label_threshold, \
                                                lstm_units, \
                                                lr, \
                                                epsilon, \
                                                regularizer, \
                                                data_set))

        # if using 2 gpu, will only display half of the tasks completed
        task['completed'][index] = 1
        task.to_csv('task.csv')



if __name__ == '__main__':
    if device_type == 'gpu':
        main(sys.argv[1])
    elif device_type == 'cpu':
        main(0)
    else:
        sys.exit("wrong device type")


