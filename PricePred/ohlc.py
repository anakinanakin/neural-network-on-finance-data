import psycopg2, psycopg2.extras
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import timedelta, date
from sklearn.metrics import confusion_matrix, classification_report

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





input_tensor = Input(shape=(30,4,1))

'''layer_x = layers.Conv2D(16, (1,1))(input_tensor)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)

layer_x = layers.Conv2D(16, (1,1))(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)
layer_x = layers.Conv2D(16, (4,1), padding='same')(layer_x)
layer_x = layers.LeakyReLU(alpha=0.01)(layer_x)'''

layer_x = layers.Conv2D(16, (1,4))(input_tensor)
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
layer_x = layers.Reshape((30,96))(layer_x)

# 64 LSTM units
layer_x = layers.LSTM(256)(layer_x)
# The last output layer uses a softmax activation function
output = layers.Dense(3, activation='softmax')(layer_x)


model = Model(input_tensor, output)
opt = Adam(lr=0.01, epsilon=1)# learning rate and epsilon are the same as paper DeepLOB 
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy']) 
#print(model.summary())




'''conn = psycopg2.connect(**eval(open('auth.txt').read()))
cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
#cmd.execute('select * from market_index where mid = 3 and dt=%(dt)s',dict(dt='2005-02-01'))
#recs = cmd.fetchall()

#df = pd.DataFrame(recs, columns = recs[0].keys())

#df['co'] = df['close']-df['open']

#change column order
#df.loc[:,['dt', 'tm', 'open', 'close', 'high', 'low']]

train_x = []
train_y = []

sample_size = 30
feature_num = 4
predict_horizon = 10

start_date = date(2005, 7, 1)
end_date = date(2005, 7, 31)
#run from start_date to end_date-1day
for single_date in date_range(start_date, end_date):
    cmd.execute('select * from market_index where mid = 3 and dt=%(dt)s',dict(dt=single_date.strftime("%Y-%m-%d")))
    recs = cmd.fetchall()
    #print(recs)
    if recs == []:
        continue;

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

    #list slicing doesn't include last element; pd.Dataframe loc does include
    #use previous 30mins to predict 10 min horizon(k=10)
    for i in df.index:
        df['horizon avg'][i] = df.loc[i+1:i+10]['close'].sum()/10.0000

    df['pct'] = (df['horizon avg']-df['close'])/df['close']

    df['target'] = 1

    #labels 0: equal or greater than 0.00015
    #labels 1: between
    #labels 2: smaller or equal to -0.00015
    df.loc[df['pct'] >= 0.0003, 'target'] = 0
    df.loc[df['pct'] <= -0.0003, 'target'] = 2

    df['target'].value_counts()

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
train_y = to_categorical(train_y)'''

save_dir = os.path.join(os.getcwd(), 'saved_data')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

train_x = np.load(os.path.join(save_dir, 'train_x_2005_01_01-06_30.npy'))
train_y = np.load(os.path.join(save_dir, 'train_y_2005_01_01-06_30_onehot.npy'))
# test_x = np.load(os.path.join(save_dir, 'test_x_2005_07_01-07_31.npy'))
# test_y = np.load(os.path.join(save_dir, 'test_y_2005_07_01-07_31_onehot.npy'))

# print(train_x.shape)
# #print(train_x)
# print(train_y.shape)
# #print(train_y)

# #np.save(os.path.join(save_dir, 'test_x_2005_07_01-07_31.npy'), train_x)
# #np.save(os.path.join(save_dir, 'test_y_2005_07_01-07_31.npy'), train_y)

save_dir = os.path.join(os.getcwd(), 'saved_models')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

filepath = "model_{epoch:02d}-{acc:.2f}.hdf5"
#save model for each epoch
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='acc',verbose=1)

# early_stopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1)



history1 = model.fit(train_x, train_y, batch_size=32, epochs=500, callbacks=[checkpoint], validation_split=0.2)



# #test_dir = os.path.join(os.getcwd(), 'test_models')
# files = glob.glob(os.path.join(save_dir, '*.hdf5'))
# files.sort(key=os.path.getmtime)

# test_loss = []
# test_acc = []


# #use too much memory
# #load model and draw each epoch's accuracy and loss
# for f in files:
#     #print(f)
#     model = load_model(f)
#     loss, acc = model.evaluate(test_x, test_y)
#     test_loss = test_loss + [loss]
#     test_acc = test_acc + [acc]



#plot train and validation accuracy
plt.plot(history1.history['acc'])
plt.plot(history1.history['val_acc'])
# plt.plot(test_acc)
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'])
plt.savefig('accuracy_lstm=256.png')
plt.clf()

#plot train and validation loss
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
# plt.plot(test_loss)
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'])
plt.savefig('loss_lstm=256.png')



#y_pred = model.predict(train_x, verbose=1)
#conf = confusion_matrix(train_y, y_pred)
#print(conf)






