import psycopg2, psycopg2.extras
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import patches
from matplotlib.pyplot import figure
from datetime import timedelta, date
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score

#error occurs when directly import keras without tensorflow.python
from tensorflow.python.keras import layers, Input, regularizers
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping



# 1. sample_size=30: the most 30 recent updates
# 2. feature_num=4: 4 features(OHLC) per time stamp
# 3. predict_horizon=10: 10 future close price, k=10
# each day don't use target label of first s-1 rows and last k rows(s = sample_size, k = predict_horizon)
def get_model_data(df, sample_size, feature_num, predict_horizon):
    dt_count = df['dt'].value_counts()
    date_num = dt_count.shape[0]
    event_num = dt_count.sum()
    input_shape = event_num-(sample_size-1+predict_horizon)
    df = df.drop(columns = ['dt'])

    data = df.values
    X = []
    Y = []
    #shape = data.shape
    #X = np.zeros((input_shape, sample_size, feature_num))
    #Y = np.zeros((input_shape, 1))
    #e.g. take feature from 0~99 row to predict target label on 99th row, take feature from 31837~31936 row to predict target label on 31936th row
    for i in range(input_shape):#range = 0~31837
        X.append(data[i:i+sample_size,0:feature_num])# [every 100 events from 31937 rows, take the first 40 columns as features]
        Y.append(data[i+sample_size-1,-1:])# [from 99~31936 rows, take the last 5 columns as labels]
    #X = X.reshape(len(X), sample_size, feature_num, 1)# add the 4th dimension: 1 channel

    return X,Y

def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

# transform array to rectangle shape
def trans2rect(arr):
    tarr = []
    trend = arr[0]
    width = 1
    day = 0
    for elm in arr[1:]:
        if elm == trend:
            width += 1
        else:
            tarr.append((trend, day, width))
            trend = elm
            day  += width
            width = 1
    tarr.append((trend, day, width))
    return tarr

def train_model(model, train_x, train_y, test_x, test_y, batch_size=32, epochs=5, validation_split=0.2):
    train_acc = []
    train_loss = []
    valid_acc = []
    valid_loss = []
    test_acc = []
    test_loss = []
    test_f1 = []
    test_precision = []
    test_recall = []

    test_y_label = [np.where(r==1)[0][0] for r in test_y]

    for i in range(epochs):
        print('starting epoch {}'.format(i+1))
        history1 = model.fit(train_x, train_y, batch_size=batch_size, epochs=1, validation_split=validation_split)

        train_acc =  train_acc  + history1.history['acc']
        train_loss = train_loss + history1.history['loss']
        valid_acc =  valid_acc  + history1.history['val_acc']
        valid_loss = valid_loss + history1.history['val_loss']

        loss, acc = model.evaluate(test_x, test_y)
        test_loss = test_loss + [loss]
        test_acc =  test_acc + [acc]
        
        y_pred = model.predict(test_x, verbose=1)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred.tolist()

        #target_names = ['increase', 'stationary', 'decrease']
        #print(classification_report(test_y_label, y_pred, target_names=target_names))
        f1 = f1_score(test_y_label, y_pred, average='macro')
        precision = precision_score(test_y_label, y_pred, average='macro')
        recall = recall_score(test_y_label, y_pred, average='macro')

        test_f1 = test_f1 + [f1]
        test_precision = test_precision + [precision]
        test_recall = test_recall + [recall]

    #print(train_acc, train_loss, valid_acc, valid_loss, test_acc, test_loss, test_f1, test_precision, test_recall)
    return train_acc, train_loss, valid_acc, valid_loss, test_acc, test_loss, test_f1, test_precision, test_recall






# input_tensor = Input(shape=(30,4,1))

# '''layer_x = layers.Conv2D(16, (1,1))(input_tensor)
# layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
# layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
# layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
# layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
# layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

# layer_x = layers.Conv2D(16, (1,1))(layer_x)
# layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
# layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
# layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
# layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
# layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)'''

# layer_x = layers.Conv2D(16, (1,4))(input_tensor)
# layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
# layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
# layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
# layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
# layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

# # Inception Module
# tower_1 = layers.Conv2D(32, (1,1), padding='same')(layer_x)
# tower_1 = layers.LeakyReLU(alpha=0.01)(tower_1)
# tower_1 = layers.Conv2D(32, (3,1), padding='same')(tower_1)
# tower_1 = layers.LeakyReLU(alpha=0.01)(tower_1)

# tower_2 = layers.Conv2D(32, (1,1), padding='same')(layer_x)
# tower_2 = layers.LeakyReLU(alpha=0.01)(tower_2)
# tower_2 = layers.Conv2D(32, (5,1), padding='same')(tower_2)
# tower_2 = layers.LeakyReLU(alpha=0.01)(tower_2)  

# tower_3 = layers.MaxPooling2D((3,1), padding='same', strides=(1,1))(layer_x)
# tower_3 = layers.Conv2D(32, (1,1), padding='same')(tower_3)
# tower_3 = layers.LeakyReLU(alpha=0.01)(tower_3)

# layer_x = layers.concatenate([tower_1, tower_2, tower_3], axis=-1)

# # concatenate features of tower_1, tower_2, tower_3
# layer_x = layers.Reshape((30,96))(layer_x)

# # 64 LSTM units
# layer_x = layers.LSTM(1)(layer_x)
# # The last output layer uses a softmax activation function
# output = layers.Dense(3, activation='softmax')(layer_x)


# model = Model(input_tensor, output)
# opt = Adam(lr=0.01, epsilon=1)# learning rate and epsilon are the same as paper DeepLOB 
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 
#print(model.summary())




conn = psycopg2.connect(**eval(open('auth.txt').read()))
cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
cmd.execute('select * from market_index where mid = 3 and dt=%(dt)s',dict(dt='2005-02-01'))
recs = cmd.fetchall()

df = pd.DataFrame(recs, columns = recs[0].keys())

df['co'] = df['close']-df['open']

#change column order
df.loc[:,['dt', 'tm', 'open', 'close', 'high', 'low']]

train_x = []
train_y = []

sample_size = 50
feature_num = 4
predict_horizon = 30
label_threshold = 0.0005

start_date = date(2010, 7, 1)
end_date = date(2010, 7, 31)
#working_day = 0
#figure(num=None, figsize=(48, 10), dpi=80, facecolor='w', edgecolor='k')

#run from start_date to end_date-1day
for single_date in date_range(start_date, end_date):
    cmd.execute('select * from market_index where mid = 3 and dt=%(dt)s',dict(dt=single_date.strftime("%Y-%m-%d")))
    recs = cmd.fetchall()
    #print(recs)
    if recs == []:
        continue;

#     working_day = working_day+1

# print(working_day)
    df = pd.DataFrame(recs, columns = recs[0].keys())

    #cmd.execute('select * from market_index where mid = 3 and dt between %(dt1)s and %(dt2)s',dict(dt1='2005-01-01', dt2='2005-01-10'))
    #len(recs)

    df.sort_values(by='dt')

    #df = df[df.origin == True]

    df = df.drop(columns = ['mid', 'tm', 'volume', 'origin'])

    #percentage change of each row
    #df['pct'] = df['close'].pct_change()
    #df['pct'] = df['pct'].shift(-1)

    df['horizon avg'] = 0.000000

    #use previous 30mins to predict 10 min horizon(k=10)

    #list slicing doesn't include last element; pd.Dataframe loc does include
    for i in df.index:
        df['horizon avg'][i] = df.loc[i+1:i+predict_horizon]['close'].sum()/float(predict_horizon)

    df['pct'] = (df['horizon avg']-df['close'])/df['close']

    df['target'] = 1

    #labels 0: equal or greater than 0.00015
    #labels 1: between
    #labels 2: smaller or equal to -0.00015
    df.loc[df['pct'] >=       label_threshold, 'target'] = 0
    df.loc[df['pct'] <=  (-1)*label_threshold, 'target'] = 2

    # label = df['target'].values.tolist()
    # ax = plt.subplot(111)
    # tans = trans2rect(label)

    # tans_stats = sorted(tans, key=lambda x: x[2])
    # for a in tans:
    #     if a[0] == 0:
    #         col = (1,.6,.6)
    #     elif a[0] == 1:
    #         col = 'w'
    #     elif a[0] == 2:
    #         col = (.6,1,.6) 

    #     ax.add_patch(patches.Rectangle((a[1],0), a[2],1, color=col))

    # close_price = df['close'].values.tolist()
    # close_price = [(float(i)-min(close_price))/(max(close_price)-min(close_price)) for i in close_price]
    # plt.plot(close_price)
    # #plt.plot(ps)
    # plt.title('date={}, k={}, threshold={}, #lables={}, max_period={}'.format(single_date, predict_horizon, label_threshold, len(tans), tans_stats[-1][2]))
    # plt.savefig('date={}_k={}_threshold={}.png'.format(single_date, predict_horizon, label_threshold*10000))
    # plt.clf()

    # print(df['target'].value_counts())

    df = df.drop(columns = ['pct', 'horizon avg'])

    #mean = (df['open'].mean()+df['high'].mean()+df['low'].mean()+df['close'].mean())/4

    df1 = df['open']
    df2 = df['high']
    df3 = df['low']
    df4 = df['close']

    df5 = pd.concat([df1, df2, df3, df4], ignore_index=True)
    mean = df5.mean()
    std = df5.std()

    #zscore
    df['open'] = (df['open']-mean)/std
    df['high'] = (df['high']-mean)/std
    df['low'] = (df['low']-mean)/std
    df['close'] = (df['close']-mean)/std

    #if single_date == start_date:
    #else:
    x, y = get_model_data(df, sample_size, feature_num, predict_horizon)
    #print(x)
    #print(y)

    #np array
    #train_x = np.append(train_x, x, axis = 0)
    #train_y = np.append(train_y, y, axis = 0)
    #list
    train_x = train_x + x
    train_y = train_y + y

train_x = np.array(train_x)
train_y = np.array(train_y)
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)


train_y = train_y.astype(int)


save_dir = os.path.join(os.getcwd(), 'saved_data/2010')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# train_x = np.load(os.path.join(save_dir, 'train_x_2005_01_01-06_30.npy'))
# train_y = np.load(os.path.join(save_dir, 'train_y_2005_01_01-06_30_onehot.npy'))
# test_x = np.load(os.path.join(save_dir, 'test_x_2005_07_01-07_31.npy'))
# test_y = np.load(os.path.join(save_dir, 'test_y_2005_07_01-07_31_onehot.npy'))

# # print(train_x.shape)
# # #print(train_x)
# # print(train_y.shape)
# # #print(train_y)

np.save(os.path.join(save_dir, 'test_x_2010_07_ohlc_price_input={}_k={}_threshold={}.npy'.format(sample_size, predict_horizon, label_threshold)), train_x)
np.save(os.path.join(save_dir, 'test_y_2010_07_ohlc_price_input={}_k={}_threshold={}.npy'.format(sample_size, predict_horizon, label_threshold)), train_y)

train_y = to_categorical(train_y)

np.save(os.path.join(save_dir, 'test_y_2010_07_ohlc_price_input={}_k={}_threshold={}_onehot.npy'.format(sample_size, predict_horizon, label_threshold)), train_y)


# save_dir = os.path.join(os.getcwd(), 'saved_models')
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)

# filepath = "model_{epoch:02d}-{acc:.2f}.hdf5"
# #save model for each epoch
# checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='acc',verbose=1)

# # # early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)





#history1 = model.fit(train_x, train_y, batch_size=32, epochs=5, callbacks=[checkpoint], validation_split=0.2)
#history1 = model.fit(train_x, train_y, batch_size=32, epochs=5, validation_split=0.2)

#model = load_model('model_100-0.58.hdf5')

#train_acc, train_loss, valid_acc, valid_loss, test_acc, test_loss, test_f1, test_precision, test_recall = train_model(model, train_x, train_y, test_x, test_y, batch_size=32, epochs=5, validation_split=0.2)


# # #test_dir = os.path.join(os.getcwd(), 'test_models')
# # files = glob.glob(os.path.join(save_dir, '*.hdf5'))
# # files.sort(key=os.path.getmtime)

# # test_loss = []
# # test_acc = []


# # #use too much memory
# # #load model and draw each epoch's accuracy and loss
# # for f in files:
# #     #print(f)
# #     model = load_model(f)
# #     loss, acc = model.evaluate(test_x, test_y)
# #     test_loss = test_loss + [loss]
# #     test_acc = test_acc + [acc]



#plot train and validation accuracy
# plt.plot(train_acc)
# plt.plot(valid_acc)
# plt.plot(test_acc)
# plt.title('Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Valid', 'Test'])
# plt.savefig('accuracy.png')
# plt.clf()

# #plot train and validation loss
# plt.plot(train_loss)
# plt.plot(valid_loss)
# plt.plot(test_loss)
# plt.title('Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Valid', 'Test'])
# plt.savefig('loss.png')
# plt.clf()

# plt.plot(test_acc)
# plt.plot(test_recall)
# plt.plot(test_precision)
# plt.plot(test_f1)
# plt.title('Test evaluation')
# plt.ylabel('Score')
# plt.xlabel('Epoch')
# plt.legend(['Accuracy', 'Recall', 'Precision', 'F1'])
# plt.savefig('test_evaluation.png')
# plt.clf()


# model = load_model('model_09-1.00.hdf5')

# test_y = [np.where(r==1)[0][0] for r in test_y]

# y_pred = model.predict(test_x, verbose=1)
# y_pred = np.argmax(y_pred, axis=1)
# y_pred.tolist()
# #conf = confusion_matrix(test_y, y_pred)
# print(test_y)
# print(y_pred)
#print(conf)


# target_names = ['increase', 'stationary', 'decrease']
# print(classification_report(test_y, y_pred, target_names=target_names))


#output directory naming

tid = 12854

#'00003236'
hex_num = '{:08x}'.format(tid)

#['00', '00', '32', '36']
hex_num = ['{:08x}'.format(tid)[i*2:i*2+2] for i in range(4)]

#'00/00/32/36'
save_dir = '/'.join(['{:08x}'.format(tid)[i*2:i*2+2] for i in range(4)])





