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





conn = psycopg2.connect(**eval(open('auth.txt').read()))
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

start_date = date(2008, 2, 1)
end_date = date(2008, 2, 5)
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

    #use previous 30mins to predict 10 min horizon(k=10)
    print(df.loc[1:10]['close'])
    #for i in df.index:
        #df['horizon avg'][i] = df.loc[i+1:i+10]['close'].sum()/10.0000


    # df['pct'] = (df['horizon avg']-df['close'])/df['close']
    # #print(df['pct'])

    # df['target'] = 1

    # #labels 0: equal or greater than 0.00015
    # #labels 1: between
    # #labels 2: smaller or equal to -0.00015
    # df.loc[df['pct'] >=  0.1, 'target'] = 0
    # df.loc[df['pct'] <= -0.1, 'target'] = 2

    # print(df['target'].value_counts())