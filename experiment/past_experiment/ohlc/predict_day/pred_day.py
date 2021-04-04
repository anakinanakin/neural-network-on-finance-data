import psycopg2, psycopg2.extras
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, recall_score, precision_score

#error occurs when directly import keras without tensorflow.python
from tensorflow.python.keras import layers, Input, regularizers
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping



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

def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def get_f1_pre_recall(model, x, y):
    y_pred = model.predict(x, verbose=2)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred.tolist()

    f1 = f1_score(y, y_pred, average='macro')
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')

    return f1, precision, recall

def get_model_data(df, input_size, feature_num, pred_k):
    dt_count = df['dt'].value_counts()
    date_num = dt_count.shape[0]
    event_num = dt_count.sum()
    input_shape = event_num-(input_size-1+pred_k)
    df = df.drop(columns = ['dt'])

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df) 

    data = df.values
    X = []
    Y = []

    for i in range(input_shape):#range = 0~31837
        X.append(data[i:i+input_size,0:feature_num])# [every 100 events from 31937 rows, take the first 40 columns as features]
        Y.append(data[i+input_size-1,-1:])# [from 99~31936 rows, take the last 5 columns as labels]

    return X,Y

def main():

    conn = psycopg2.connect(**eval(open('auth.txt').read()))
    cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    start_date = date(2010, 7, 1)
    end_date = date(2010, 7, 2)

    input_size = 30
    pred_k = 50
    feature_num = 4
    label_threshold = 0.0010

    #run from start_date to end_date-1 day
    for single_date in date_range(start_date, end_date):

        test_x = []
        test_y = []

        cmd.execute('select * from market_index where mid = 1 and dt=%(dt)s',dict(dt=single_date.strftime("%Y-%m-%d")))
        recs = cmd.fetchall()

        if recs == []:
            continue;

        df = pd.DataFrame(recs, columns = recs[0].keys())

        #change column order
        df.loc[:,['dt', 'tm', 'open', 'close', 'high', 'low', 'volume']]

        df.sort_values(by='dt')

        
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

        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(df) 


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

        x, y = get_model_data(df, input_size, feature_num, pred_k)

        #list
        test_x = test_x + x
        test_y = test_y + y


        test_x = np.array(test_x)
        test_y = np.array(test_y)
        test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2], 1)


        test_y = test_y.astype(int)

        temp = []
        test_y = [i[0] for i in test_y]

        # print(len(test_y))
        # print(test_y)

        # save_dir = os.path.join(os.getcwd(), 'data_set/'+str(data_set))
        # if not os.path.isdir(save_dir):
        #     os.makedirs(save_dir)

        #np.save(os.path.join(save_dir, 'valid_x.npy'), test_x)

        #test_y = to_categorical(test_y)

        #np.save(os.path.join(save_dir, 'valid_y_onehot.npy'), test_y)

        model = load_model('model_epoch_300.h5')

        y_pred = model.predict(test_x, verbose=2)
        y_pred = np.argmax(y_pred, axis=1)
        y_pred.tolist()

        acc = accuracy_score(test_y, y_pred)

        # print(len(y_pred))
        # print(y_pred)

        #loss, acc = model.evaluate(test_x, test_y, verbose=2)
        # test_loss = test_loss + [loss]
        # test_acc =  test_acc + [acc]

        # f1, precision, recall = get_f1_pre_recall(model, test_x, test_y_label)

        # test_f1 = test_f1 + [f1]
        # test_precision = test_precision + [precision]
        # test_recall = test_recall + [recall]

        plt.rcParams.update({'font.size': 35})
        figure(figsize=(100,40), dpi=80)
        
        plt.suptitle('date={}, acc={}, k={}, threshold={}'.format(single_date, acc, pred_k, label_threshold))
        

        ax = plt.subplot(211)

        tans = trans2rect(test_y)
        print(len(test_y))
        print(test_y)
        tans_stats = sorted(tans, key=lambda x: x[2])

        plt.title('Answer, #lables={}, max_period={}'.format(len(tans), tans_stats[-1][2]))
        

        
        for a in tans:
            if a[0] == 0:
                col = (1,.6,.6)
            elif a[0] == 1:
                col = 'w'
            elif a[0] == 2:
                col = (.6,1,.6) 

            ax.add_patch(patches.Rectangle((a[1],0), a[2],1, color=col))

        df_close = df4
        close_price = df_close.values.tolist()
        close_price = [(float(i)-min(close_price))/(max(close_price)-min(close_price)) for i in close_price]
        close_price = close_price[input_size-1:-pred_k]
        
        plt.plot(close_price)


        ax = plt.subplot(212)

        tans = trans2rect(y_pred)
        tans_stats = sorted(tans, key=lambda x: x[2])

        plt.title('Prediction, #lables={}, max_period={}'.format(len(tans), tans_stats[-1][2]))
        

        
        for a in tans:
            if a[0] == 0:
                col = (1,.6,.6)
            elif a[0] == 1:
                col = 'w'
            elif a[0] == 2:
                col = (.6,1,.6) 

            ax.add_patch(patches.Rectangle((a[1],0), a[2],1, color=col))
        
        plt.plot(close_price)


        plt.savefig('date={}_acc={:.2f}_k={}_threshold={}.png'.format(single_date, acc, pred_k, label_threshold*10000))



if __name__ == '__main__':
    main()
                 




