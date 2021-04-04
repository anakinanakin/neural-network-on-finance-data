import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# # fix result, otherwise may predict all flat
# np.random.seed(1271)
# tf.compat.v1.set_random_seed(1271)

from matplotlib import patches
from matplotlib.pyplot import figure
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score, log_loss

#error occurs when directly import keras without tensorflow.python
from tensorflow.python.keras import layers, Input, regularizers
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping



def get_f1_pre_recall(model, x, y, batch_size):
    y_pred = model.predict(x, verbose=2, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred.tolist()

    f1 = f1_score(y, y_pred, average='macro')
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')

    return f1, precision, recall

def train_model(model, save_dir, train_x, train_y, valid_x, valid_y, batch_size=32, epochs=500):

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
        print('starting epoch {}/{}'.format(i+1, epochs))
        history1 = model.fit(train_x, train_y, batch_size=batch_size, epochs=1, validation_data=(valid_x, valid_y), verbose=2)

        if (i+1)%50==0:
            #model_dir = 'result/{}/'.format(task_id)+'model/model_epoch_{}.h5'.format(i+1)
            model_dir = 'result/model/model_epoch_{}.h5'.format(i+1)
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


        if (i+1)%50==0:
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

def deeplob_model():
    input_tensor = Input(shape=(100,20,1))

    # convolutional filter is (1,2) with stride of (1,2)
    layer_x = layers.Conv2D(16, (1,2), strides=(1,2))(input_tensor)
    layer_x = layers.BatchNormalization()(layer_x)
    layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
    layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
    layer_x = layers.BatchNormalization()(layer_x)
    layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
    layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
    layer_x = layers.BatchNormalization()(layer_x)
    layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

    layer_x = layers.Conv2D(16, (1,2), strides=(1,2))(layer_x)
    layer_x = layers.BatchNormalization()(layer_x)
    layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
    layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
    layer_x = layers.BatchNormalization()(layer_x)
    layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
    layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
    layer_x = layers.BatchNormalization()(layer_x)
    layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

    layer_x = layers.Conv2D(16, (1,5))(layer_x)
    layer_x = layers.BatchNormalization()(layer_x)
    layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
    layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
    layer_x = layers.BatchNormalization()(layer_x)
    layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
    layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
    layer_x = layers.BatchNormalization()(layer_x)
    layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

    # Inception Module
    tower_1 = layers.Conv2D(32, (1,1), padding='same')(layer_x)
    layer_x = layers.BatchNormalization()(layer_x)
    tower_1 = layers.LeakyReLU(alpha=0.01)(tower_1)
    tower_1 = layers.Conv2D(32, (3,1), padding='same')(tower_1)
    layer_x = layers.BatchNormalization()(layer_x)
    tower_1 = layers.LeakyReLU(alpha=0.01)(tower_1)

    tower_2 = layers.Conv2D(32, (1,1), padding='same')(layer_x)
    layer_x = layers.BatchNormalization()(layer_x)
    tower_2 = layers.LeakyReLU(alpha=0.01)(tower_2)
    tower_2 = layers.Conv2D(32, (5,1), padding='same')(tower_2)
    layer_x = layers.BatchNormalization()(layer_x)
    tower_2 = layers.LeakyReLU(alpha=0.01)(tower_2)  

    tower_3 = layers.MaxPooling2D((3,1), padding='same', strides=(1,1))(layer_x)
    tower_3 = layers.Conv2D(32, (1,1), padding='same')(tower_3)
    layer_x = layers.BatchNormalization()(layer_x)
    tower_3 = layers.LeakyReLU(alpha=0.01)(tower_3)

    layer_x = layers.concatenate([tower_1, tower_2, tower_3], axis=-1)

    # concatenate features of tower_1, tower_2, tower_3
    layer_x = layers.Reshape((100,96))(layer_x)

    # 64 LSTM units
    #CPU version
    #layer_x = layers.LSTM(64)(layer_x)
    #GPU version, cannot run on CPU
    layer_x = layers.CuDNNLSTM(64)(layer_x)
    # The last output layer uses a softmax activation function
    output = layers.Dense(3, activation='softmax')(layer_x)

    model = Model(input_tensor, output)
    opt = Adam(lr=0.01, epsilon=1)# learning rate and epsilon are the same as paper DeepLOB 
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 

    return model

def main():
    load_dir = os.path.join(os.getcwd(), 'data_set')
    if not os.path.isdir(load_dir):
        sys.exit("no data set directory")

    train_x = np.load(load_dir+'/train_x.npy')
    train_y = np.load(load_dir+'/train_y_onehot.npy')

    valid_x = np.load(load_dir+'/valid_x.npy')
    valid_y = np.load(load_dir+'/valid_y_onehot.npy')

    save_dir = os.path.join(os.getcwd(), 'result')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)  

    model = deeplob_model()      

    train_model(model, save_dir, train_x, train_y, valid_x, valid_y, batch_size=512, epochs=300)


if __name__ == '__main__':
    main()















