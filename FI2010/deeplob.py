import psycopg2, psycopg2.extras
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import patches
from matplotlib.pyplot import figure
from IPython.display import SVG
from sklearn.metrics import classification_report

#error occurs when directly import keras without tensorflow.python
from tensorflow.python.keras import layers, Input, regularizers
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping




# Import data:
# 1. read the text file line by line;
# 2. format the data in DataFrame.

def read_data(path):
    data_list = []
    with open(path, 'rb') as f:
        while True:
            line = f.readline()
            if not line:
                break
            d_str = line.split()
            try:
                d_tem = [float(d) for d in d_str]
            except ValueError:
                pass
            data_list.append(d_tem)
    data = pd.DataFrame(data_list)
    return data.T

# ready data for training:
# 1. sample_size=100: the most 100 recent updates
# 2. feature_num=40: 40 features per time stamp
# 3. target_num=5: relative changes for the next 1,2,3,5 and 10 events(5 in total), 
#	  			   using equation 3 in the paper to calculate average future midprice and label the price movements as 0,1,2
def get_model_data(data, sample_size=100, feature_num=40, target_num=5):
    data = data.values
    shape = data.shape
    X = np.zeros((shape[0]-sample_size, sample_size, feature_num))
    Y = np.zeros((shape[0]-sample_size, target_num))
    #e.g. take feature from 0~99 row to predict target label on 99th row, take feature from 31837~31936 row to predict target label on 31936th row
    for i in range(shape[0]-sample_size):#range = 0~31837
        X[i] = data[i:i+sample_size,0:feature_num]# [every 100 events from 31937 rows, take the first 40 columns as features]
        Y[i] = data[i+sample_size-1,-target_num:]# [from 99~31936 rows, take the last 5 columns as labels]
    X = X.reshape(X.shape[0], sample_size, feature_num, 1)# add the 4th dimension: 1 channel
    
    # "Benchmark dataset for mid-price forecasting of limit order book data with machine learning"
    # Y=Y-1 relabels as 0,1,2
    # labels 0: equal to or greater than 0.002
    # labels 1: -0.00199 to 0.00199
    # labels 2: smaller or equal to -0.002
    Y = Y-1
    return X,Y

# transform array to rectangle shape
# def trans2rect(arr):
# 	tarr = []
# 	trend = arr[0]
# 	width = 1
# 	day = 0
# 	for elm in arr[1:]:
# 		if elm == trend:
# 			width += 1
# 		else:
# 			tarr.append((trend, day, width))
# 			trend = elm
# 			day  += width
# 			width = 1
# 	tarr.append((trend, day, width))
# 	return tarr

# # callback for evaluating on each epoch
# class EvaluateCallback(Callback):
#     def __init__(self, test_x, test_y, list_loss, list_acc):
#         self.test_x = test_x
#         self.test_y = test_y
#         self.list_loss = list_loss
#         self.list_acc = list_acc

#     def on_epoch_end(self, epoch, logs={}):
#         loss, acc = self.model.evaluate(self.test_x, self.test_y, verbose=0)
#         self.list_loss.append(loss)
#         self.list_acc.append(acc)






# # the size of a single input is (100,40)
input_tensor = Input(shape=(100,40,1))

# convolutional filter is (1,2) with stride of (1,2)
layer_x = layers.Conv2D(16, (1,2), strides=(1,2))(input_tensor)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

layer_x = layers.Conv2D(16, (1,2), strides=(1,2))(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

layer_x = layers.Conv2D(16, (1,10))(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

# Inception Module
tower_1 = layers.Conv2D(32, (1,1), padding='same')(layer_x)
tower_1 = layers.LeakyReLU(alpha=0.01)(tower_1)
tower_1 = layers.Conv2D(32, (3,1), padding='same')(tower_1)
tower_1 = layers.LeakyReLU(alpha=0.01)(tower_1)

tower_2 = layers.Conv2D(32, (1,1), padding='same')(layer_x)
tower_2 = layers.LeakyReLU(alpha=0.01)(tower_2)
tower_2 = layers.Conv2D(32, (5,1), padding='same')(tower_2)
tower_2 = layers.LeakyReLU(alpha=0.01)(tower_2)  

tower_3 = layers.MaxPooling2D((3,1), padding='same', strides=(1,1))(layer_x)
tower_3 = layers.Conv2D(32, (1,1), padding='same')(tower_3)
tower_3 = layers.LeakyReLU(alpha=0.01)(tower_3)

layer_x = layers.concatenate([tower_1, tower_2, tower_3], axis=-1)

# concatenate features of tower_1, tower_2, tower_3
layer_x = layers.Reshape((100,96))(layer_x)

# 64 LSTM units
#CPU version
layer_x = layers.LSTM(64)(layer_x)
#GPU version, cannot run on CPU
#layer_x = layers.CuDNNLSTM(64)(layer_x)
# The last output layer uses a softmax activation function
output = layers.Dense(3, activation='softmax')(layer_x)

model = Model(input_tensor, output)
opt = Adam(lr=0.01, epsilon=1)# learning rate and epsilon are the same as paper DeepLOB 
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 
print(model.summary())

data_path = os.path.join(os.getcwd(), 'Train_Dst_NoAuction_ZScore_CF_9.txt')
data = read_data(data_path)
train_X, train_Y = get_model_data(data)
train_Y = train_Y.astype(int)

# #separate 5 target variables(next 1,2,3,5 and 10 events)
# train_y = to_categorical(train_Y[:,0])# y1 is the next event's mid price (k=1) 
# train_y = to_categorical(train_Y[:,1])# k=2 
# train_y = to_categorical(train_Y[:,2])# k=3 
# train_y = to_categorical(train_Y[:,3])# k=5 
train_y = to_categorical(train_Y[:,4])# k=10

# #test_data.to_csv('FI2010_test.csv')
# test_path = os.path.join(os.getcwd(), 'BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt')
# #test_data.shape = 31937x149 (31937 events, 149 features)
# test_data = read_data(test_path)
# #test_X.shape = 31837x100x40x1 (31837 time frames, each with 100 events, each event with 40 features, 1 channel)
# test_X, test_Y = get_model_data(test_data)
# #test_Y.shape = 31837x5(5 target variables)
# test_Y = test_Y.astype(int)

# data = test_data.values
# midprice = data[:, 41]
# midprice = midprice[:200]

#test_y.shape = 31837x3(one hot encoding: 1,0,0; 0,1,0; 0,0,1)
# test_y = to_categorical(test_Y[:,0])
# test_y = to_categorical(test_Y[:,1])
# test_y = to_categorical(test_Y[:,2])
# test_y = to_categorical(test_Y[:,3])
# test_y = to_categorical(test_Y[:,4])

save_dir = os.path.join(os.getcwd(), 'saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

filepath="model_{epoch:02d}-{acc:.2f}.hdf5"
#save model for each epoch
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='acc',verbose=1)

# test_loss = []
# test_acc = []

# #evaluate on each epoch
# evaluate_callback = EvaluateCallback(test_X, test_y1, test_loss, test_acc)

#no difference between h5 & hdf5
# model = load_model('model1-with-70epochs.h5')

history1 = model.fit(train_X, train_y, epochs=1, batch_size=32, callbacks=[checkpoint], validation_split=0.2)

#print(history1.history.keys())#['loss', 'acc']

# results = model.evaluate(test_X, test_y1)
# print(model.metrics_names)
# # print(results)

# y_pred1 = model.predict(test_X, verbose=1)
# #y_pred2 = model2.predict(test_X, batch_size=32, verbose=1)
# #y_pred3 = model3.predict(test_X, batch_size=32, verbose=1)
# #y_pred5 = model5.predict(test_X, batch_size=32, verbose=1)
# #y_pred10 = model10.predict(test_X, batch_size=32, verbose=1)

# y_pred1 = np.argmax(y_pred1, axis=1)
# y_pred1.tolist()

#test_y1 = [np.where(r==1)[0][0] for r in test_y1]
# target_names = ['rise', 'stable', 'fall']
# print(classification_report(test_y1, y_pred1, target_names=target_names))

#plot train and validation accuracy
# plt.plot(history1.history['acc'])
# plt.plot(history1.history['val_acc'])
# # plt.plot(test_acc)
# plt.title('Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Valid'])
# plt.savefig('accuracy_k=10.png')
# plt.clf()

# #plot train and validation loss
# plt.plot(history1.history['loss'])
# plt.plot(history1.history['val_loss'])
# # plt.plot(test_loss)
# plt.title('Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Valid'])
# plt.savefig('loss_k=10.png')

# #plot rectangle graph
# figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
# ax = plt.subplot(111)
# #plt.xlim(0, 100)

# # y_pred1 = y_pred1[:200]
# test_y1 = test_y1[:200]

# tans = trans2rect(test_y1)
# # tpred = trans2rect(y_pred1)

# #ans at top, pred at bottom
# #label 0:rise, color=red
# #label 1:stable, color=white
# #label 2:fall, color=green

# for a in tans:
#     if a[0] == 0:
#         col = (1,.6,.6)
#     elif a[0] == 1:
#         col = 'w'
#     elif a[0] == 2:
#         col = (.6,1,.6)

#     ax.add_patch(patches.Rectangle((a[1],0), a[2],3, color=col))


# # for a in tpred:
# #     if a[0] == 0:
# #         col = (1,.6,.6)
# #     elif a[0] == 1:
# #         col = 'w'
# #     elif a[0] == 2:
# #         col = (.6,1,.6)
        
# #     ax.add_patch(patches.Rectangle((a[1],0), a[2],1.5, color=col))

# plt.plot(midprice)
# #save before show, otherwise can't save after closing show window
# plt.savefig('label_1.png')
#plt.show()

















