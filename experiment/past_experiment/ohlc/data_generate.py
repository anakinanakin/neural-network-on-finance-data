import psycopg2, psycopg2.extras
import os
import glob
import csv
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import patches
from matplotlib.pyplot import figure
from datetime import timedelta, date
from itertools import product
from tensorflow.python.keras.utils import to_categorical



# 1. input_size=30: the most 30 recent updates
# 2. feature_num=4: 4 features(OHLC) per time stamp
# 3. pred_k=10: 10 future close price, k=10
# each day don't use target label of first s-1 rows and last k rows(s = input_size, k = pred_k)
def get_model_data(df, input_size, feature_num, pred_k):
    dt_count = df['dt'].value_counts()
    date_num = dt_count.shape[0]
    event_num = dt_count.sum()
    input_shape = event_num-(input_size-1+pred_k)
    df = df.drop(columns = ['dt'])

    data = df.values
    X = []
    Y = []
    #shape = data.shape
    #X = np.zeros((input_shape, input_size, feature_num))
    #Y = np.zeros((input_shape, 1))
    #e.g. take feature from 0~99 row to predict target label on 99th row, take feature from 31837~31936 row to predict target label on 31936th row
    for i in range(input_shape):#range = 0~31837
        X.append(data[i:i+input_size,0:feature_num])# [every 100 events from 31937 rows, take the first 40 columns as features]
        Y.append(data[i+input_size-1,-1:])# [from 99~31936 rows, take the last 5 columns as labels]
    #X = X.reshape(len(X), input_size, feature_num, 1)# add the 4th dimension: 1 channel

    return X,Y

def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)



input_list = [30,50,100]
pred_list = [30,50]
feature_list = [4,5]
threshold_list = [0.0002,0.0003,0.0004,0.0006,0.0008,0.0010]
lstm_list = [16,32,64]
lr_list = [0.01,0.001]
regularizer_list = [0,0.001]

data_set_num = len(lstm_list)*len(lr_list)*len(regularizer_list)


data_list = [[a,b,c,d,e,f,g] for a,b,c,d,e,f,g in product(input_list, pred_list, feature_list, threshold_list, lstm_list, lr_list, regularizer_list)]

hyper_df = pd.DataFrame(np.array(data_list), columns=['input', 'k', 'feature_num', 'label_threshold',  'lstm_units', 'learning_rate', 'regularizer'])

#hyper_df['data_set'] = hyper_df.index+1

hyper_df = hyper_df.sort_values(by=['input', 'k', 'feature_num', 'label_threshold', 'lstm_units', 'learning_rate', 'regularizer'])

hyper_df = hyper_df.reset_index(drop=True)

hyper_df['data_set'] = hyper_df.index/data_set_num + 1

hyper_df['task_id'] = hyper_df.index+1

hyper_df = hyper_df[['task_id', 'input', 'k', 'feature_num', 'label_threshold', 'lstm_units', 'learning_rate', 'regularizer', 'data_set']]

#hyper_df = hyper_df.iloc[[6]]

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print(hyper_df) 

# hyper_df.to_csv('task.csv', index=False)



conn = psycopg2.connect(**eval(open('auth.txt').read()))
cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

start_date = date(2010, 1, 1)
end_date = date(2010, 7, 1)

data_set_copy = 0

#start_time = time.time()


for index, row in hyper_df.iterrows():
    data_set = int(row['data_set'])

    if data_set_copy == data_set:
        continue
    data_set_copy = data_set

    print('starting train data_set:{}'.format(data_set))

    save_dir = os.path.join(os.getcwd(), 'data_set/'+str(data_set))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if os.path.isfile(os.path.join(save_dir, 'train_y_onehot.npy')):
        continue

    train_x = []
    train_y = []

    input_size = int(row['input'])
    pred_k = int(row['k'])
    feature_num = int(row['feature_num'])
    label_threshold = float(row['label_threshold'])

    #run from start_date to end_date-1 day
    for single_date in date_range(start_date, end_date):
        #smp no volume
        #cmd.execute('select * from market_index where mid = 3 and dt=%(dt)s',dict(dt=single_date.strftime("%Y-%m-%d")))

        #smp with volume
        cmd.execute('select * from market_index where mid = 1 and dt=%(dt)s',dict(dt=single_date.strftime("%Y-%m-%d")))
        recs = cmd.fetchall()

        if recs == []:
            continue;

        df = pd.DataFrame(recs, columns = recs[0].keys())

        #change column order
        df.loc[:,['dt', 'tm', 'open', 'close', 'high', 'low', 'volume']]

        df.sort_values(by='dt')

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(df) 

        df = df.drop(columns = ['mid', 'tm', 'origin'])

        if feature_num == 4:
            df = df.drop(columns = ['volume'])

        df['horizon avg'] = 0.000000

        #list slicing doesn't include last element; pd.Dataframe loc does include
        for i in df.index:
            df.loc[i,'horizon avg'] = df.loc[i+1:i+pred_k]['close'].sum()/float(pred_k)


        df['pct'] = (df['horizon avg']-df['close'])/df['close']

        df['target'] = 1

        #labels 0: equal or greater than 0.00015
        #labels 1: between
        #labels 2: smaller or equal to -0.00015
        df.loc[df['pct'] >=       label_threshold, 'target'] = 0
        df.loc[df['pct'] <=  (-1)*label_threshold, 'target'] = 2

        df = df.drop(columns = ['pct', 'horizon avg'])

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

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(df) 

        if feature_num == 5:
            mean = df['volume'].mean()
            std = df['volume'].std()

            df['volume'] = (df['volume']-mean)/std

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(df) 

        x, y = get_model_data(df, input_size, feature_num, pred_k)

        #list
        train_x = train_x + x
        train_y = train_y + y



    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
    train_y = train_y.astype(int)

    # np.save(os.path.join(save_dir, 'train_x.npy'), train_x)
    # np.save(os.path.join(save_dir, 'valid_x.npy'), train_x)
    np.save(os.path.join(save_dir, 'train_x.npy'), train_x)

    # don't need
    # np.save(os.path.join(save_dir, 'train_y.npy'), train_y)
    # np.save(os.path.join(save_dir, 'valid_y.npy'), valid_y)
    # np.save(os.path.join(save_dir, 'test_y.npy'), test_y)

    train_y = to_categorical(train_y)
    # valid_y = to_categorical(valid_y)
    # test_y = to_categorical(test_y)

    # np.save(os.path.join(save_dir, 'test_y_onehot.npy'), train_y)
    # np.save(os.path.join(save_dir, 'test_y_onehot.npy'), train_y)
    np.save(os.path.join(save_dir, 'train_y_onehot.npy'), train_y)
                    

    # with open(os.path.join(save_dir, 'readme.txt'), 'w') as f:
    #     f.write('input size = {}\nprediction k = {}\nfeature = {}\nlabel threshold = {}'.format(input_size, pred_k, feature_num, label_threshold))


#print("--- %s seconds ---" % (time.time() - start_time))