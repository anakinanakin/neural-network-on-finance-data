import psycopg2, psycopg2.extras
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import patches
from matplotlib.pyplot import figure
from datetime import timedelta, date




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





conn = psycopg2.connect(**eval(open('auth.txt').read()))
cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
cmd.execute('select * from market_index where mid = 3 and dt=%(dt)s',dict(dt='2005-02-01'))
recs = cmd.fetchall()

df = pd.DataFrame(recs, columns = recs[0].keys())

df['co'] = df['close']-df['open']

#change column order
df.loc[:,['dt', 'tm', 'open', 'close', 'high', 'low']]



predict_horizon = 10
label_threshold = 0.0007

start_date = date(2010, 6, 7)
end_date = date(2010, 6, 9)

figure(num=None, figsize=(48, 10), dpi=80, facecolor='w', edgecolor='k')

#run from start_date to end_date-1day
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
        df.loc[i,'horizon avg'] = df.loc[i+1:i+predict_horizon]['close'].sum()/float(predict_horizon)

    df['pct'] = (df['horizon avg']-df['close'])/df['close']

    df['target'] = 1

    #labels 0: equal or greater than 0.00015
    #labels 1: between
    #labels 2: smaller or equal to -0.00015
    df.loc[df['pct'] >=       label_threshold, 'target'] = 0
    df.loc[df['pct'] <=  (-1)*label_threshold, 'target'] = 2

    label = df['target'].values.tolist()
    label = label[:-predict_horizon]

    ax = plt.subplot(111)
    tans = trans2rect(label)

    tans_stats = sorted(tans, key=lambda x: x[2])
    for a in tans:
        if a[0] == 0:
            col = (1,.6,.6)
        elif a[0] == 1:
            col = 'w'
        elif a[0] == 2:
            col = (.6,1,.6) 

        ax.add_patch(patches.Rectangle((a[1],0), a[2],1, color=col))

    close_price = df['close'].values.tolist()
    close_price = [(float(i)-min(close_price))/(max(close_price)-min(close_price)) for i in close_price]
    close_price = close_price[:-predict_horizon]
    
    plt.plot(close_price)
    #plt.plot(ps)
    plt.title('date={}, k={}, threshold={}, #lables={}, max_period={}'.format(single_date, predict_horizon, label_threshold, len(tans), tans_stats[-1][2]))
    plt.savefig('date={}_k={}_threshold={}.png'.format(single_date, predict_horizon, label_threshold*10000))
    plt.clf()

    