import psycopg2, psycopg2.extras
import os
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import patches
from matplotlib.pyplot import figure
from datetime import timedelta, date



def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)




conn = psycopg2.connect(**eval(open('auth.txt').read()))
cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)


pred_list = [30]
threshold_list = [0.0016]
#threshold_list = [0.0004, 0.0007, 0.001, 0.0014, 0.0016, 0.002, 0.0022, 0.0024]

start_date = date(2010, 3, 22)
end_date = date(2010, 3, 23)



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

        ax = plt.subplot(111)
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
        #close_price = [(float(i)-min(close_price))/(max(close_price)-min(close_price)) for i in close_price]
        close_price = close_price[:-predict_horizon]
        
        # plt.plot(close_price)
        # #plt.plot(ps)
        # plt.title('date={}, k={}, threshold={}, #lables={}, max_period={}'.format(single_date, predict_horizon, label_threshold, len(tans), tans_stats[-1][2]))
        # plt.savefig('date={}_k={}_threshold={}.png'.format(single_date, predict_horizon, label_threshold*10000))
        # plt.clf()

        calc_profit(close_price, label)


        last_price = 0
        current_price = 0
        profit = 0
        trans_ctr = 0
        state = 0

        # 0:init state
        # 1:up state
        # -1:down state
        for p, l in zip(close_price, label):
            if state == 1:
                if l == -1:
                    current_price = p
                    profit = profit + abs(current_price-last_price)
                    trans_ctr = trans_ctr+1
                    last_price = current_price
                    state = -1

            elif state == -1:
                if l == 1:
                    current_price = p
                    profit = profit + abs(current_price-last_price)
                    trans_ctr = trans_ctr+1
                    last_price = current_price
                    state = 1

            else:
                if l == -1:
                    last_price = p
                    state = -1
                elif l == 1:
                    last_price = p
                    state = 1


        print(single_date)
        print('pred_k:{}'.format(predict_horizon))
        print('threshold:{}'.format(label_threshold))
        print('profit:'+str(profit))
        print('#trades:'+str(trans_ctr))
        if trans_ctr == 0:
            none_zero_trans_ctr = 1
        else:
            none_zero_trans_ctr = trans_ctr
        print('avg profit:'+str(profit/float(none_zero_trans_ctr)))

        print('profit-1bp:'+str(profit-trans_ctr))
        print('\n')




    

    