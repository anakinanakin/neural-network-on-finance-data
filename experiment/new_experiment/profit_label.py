import psycopg2, psycopg2.extras
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import patches
from matplotlib.pyplot import figure
from datetime import timedelta, date
from tensorflow.python.keras.utils import to_categorical





task = pd.read_csv("task.csv") 


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

def label_trans(label):

    new_label = []
    state = 0

    for l in label:
        if state == 1:
            if l == -1:
                state = -1
                new_label = new_label + [-1]
            else:
                new_label = new_label + [1]

        elif state == -1:
            if l == 1:
                state = 1
                new_label = new_label + [1]
            else:
                new_label = new_label + [-1]

        else:
            if l == -1:
                state = -1
                new_label = new_label + [-1]
            elif l == 1:
                state = 1
                new_label = new_label + [1]
            else:
                new_label = new_label + [0]

    # print(len(label))
    # print(len(new_label))

    return new_label

def trading_label():
    conn = psycopg2.connect(**eval(open('auth.txt').read()))
    cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    predict_horizon = 10
    label_threshold = 0.0012

    start_date = date(2010, 7, 1)
    end_date = date(2010, 7, 2)
    train_y = []
	
    for single_date in date_range(start_date, end_date):
    	cmd.execute('select * from market_index where mid = 1 and dt=%(dt)s',dict(dt=single_date.strftime("%Y-%m-%d")))
    	recs = cmd.fetchall()

        if recs == []:
            continue;

        df = pd.DataFrame(recs, columns = recs[0].keys())

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
            # slow
            #df['horizon avg'][i] = df.loc[i+1:i+predict_horizon]['close'].sum()/float(predict_horizon)
            df.loc[i,'horizon avg'] = df.loc[i+1:i+predict_horizon]['close'].sum()/float(predict_horizon)

        df['pct'] = (df['horizon avg']-df['close'])/df['close']

        df['target'] = 0

        #labels 1: equal or greater than 0.00015
        #labels 0: between
        #labels -1: smaller or equal to -0.00015
        df.loc[df['pct'] >=       label_threshold, 'target'] = 1
        df.loc[df['pct'] <=  (-1)*label_threshold, 'target'] = -1

        label = df['target'].values.tolist()
        label = label[:-predict_horizon]




        figure(figsize=(80,32), dpi=80)
        plt.rcParams.update({'font.size': 30})
        plt.suptitle('date={}, k={}, threshold={}'.format(single_date, predict_horizon, label_threshold))



        ax = plt.subplot(211)

        tans = trans2rect(label)
        tans_stats = sorted(tans, key=lambda x: x[2])


        for a in tans:
            if a[0] == 1:
                col = (1,.6,.6)
            elif a[0] == 0:
                col = (1,1,.6)
            elif a[0] == -1:
                col = (.6,1,.6) 

            ax.add_patch(patches.Rectangle((a[1],0), a[2],1, color=col))

        close_price = df['close'].values.tolist()
        close_price = [(float(i)-min(close_price))/(max(close_price)-min(close_price)) for i in close_price]
        close_price = close_price[:-predict_horizon]
        
        plt.plot(close_price)
        plt.title('#lables={}, max_period={}'.format(len(tans), tans_stats[-1][2]))


        new_label = label_trans(label)

        ax2 = plt.subplot(212)

        tans = trans2rect(new_label)
        tans_stats = sorted(tans, key=lambda x: x[2])

        for a in tans:
            if a[0] == 1:
                col = (1,.6,.6)
            elif a[0] == 0:
                col = (1,1,.6)
            elif a[0] == -1:
                col = (.6,1,.6) 

            ax2.add_patch(patches.Rectangle((a[1],0), a[2],1, color=col))

        
        plt.plot(close_price)
        plt.title('#lables={}, max_period={}'.format(len(tans), tans_stats[-1][2]))
        plt.savefig('date={}_k={}_threshold={}.png'.format(single_date, predict_horizon, label_threshold*10000))
        plt.clf()

        train_y = train_y + new_label


    #make every label state exist for one hot encoding
    train_y = train_y + [-1,0,1]
    print(train_y)
    
    np1 = np.ones(len(train_y))

    train_y = np.array(train_y)
    train_y = np.add(train_y, np1)
    train_y = train_y.astype(int)
    train_y = to_categorical(train_y)
    print(train_y)
    print(train_y.shape)
    train_y = np.delete(train_y, np.s_[-3:], axis = 0)
    print(train_y.shape)
    print(train_y)

    train_y = np.delete(train_y, np.s_[:99], axis = 0)
    print(train_y.shape)
    print(train_y)
    #np.save('trading_valid_y_onehot.npy', train_y)



if __name__ == '__main__':
    trading_label()