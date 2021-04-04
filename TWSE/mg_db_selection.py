# coding=utf-8

"""
def readme():

    {
    '_id': ObjectId('5ecb4b9accc7c143a38dee72')
    'DspDatetime': datetime.datetime(1992, 1, 4, 9, 0, 2, 200000)
    'Code': '1101'
    'Remark': 'A'
    'TrendFlag': ' '
    'MatchFlag': 'Y'
    'TradeUpLowLimit': ' '
    'Trade_Price': 6950
    'TxVol': 111
    'BuyTickSize': 1
    'BuyUpLowLimit': ' '
    'BuyPV5': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    'SellTickSize': 1
    'SellUpLowLimit': ' '
    'SellPV5': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    'Matcher': '45'
    }

    ### clode list

    台　泥   (1101)
    亞　泥   (1102)
    統　一   (1216)
    台　塑   (1301)
    南　亞   (1303)
    台　化   (1326)
    遠東新   (1402)
    中　鋼   (2002)
    正　新   (2105)
    和泰車   (2207)
    裕日車   (2227)
    光寶科   (2301)
    聯　電   (2303)
    台達電   (2308)
    鴻　海   (2317)
    國　巨   (2327)
    台積電   (2330)
    佳世達   (2352)
    華　碩   (2357)
    廣　達   (2382)
    研　華   (2395)
    南亞科   (2408)
    中華電   (2412)
    聯發科   (2454)
    可　成   (2474)
    陽　明   (2609)
    華　航   (2610)
    台灣高鐵 (2633)
    彰　銀   (2801)
    中　壽   (2823)
    TW50    (0050)
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymongo as mg

from datetime import datetime, timedelta
from matplotlib import patches



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

# draw trade price and mid price
def draw_price():
    client = mg.MongoClient('your mongodb client')
    db = client['twse']
    col = db['tw50']
    code = '2330'

    start_date = datetime(2019, 1, 1)
    end_date = datetime(2019, 1, 10)

    for dt in date_range(start_date, end_date):

        docs = col.find( { 'Code': code, 'DspDatetime': { '$gte': dt, '$lt':  dt+timedelta(days=1) } } )

        docs = [doc for doc in docs]

        if docs == []:
            continue


        print(len(docs))

        midprice_list = []

        for doc in docs:
            buypv5 = np.array(doc['BuyPV5']).flatten().tolist()
            sellpv5 = np.array(doc['SellPV5']).flatten().tolist()
            midprice = (buypv5[0]+sellpv5[0])/2
            print(buypv5+sellpv5, doc['Trade_Price'], midprice)

            midprice_list = midprice_list+[midprice]



        df = pd.DataFrame(docs)
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(df) 

        #draw trade price and mid price
        plt.figure(figsize=(40,16), dpi=80)
        plt.rcParams.update({'font.size': 25})

        trade_price = list(df['Trade_Price'])
        #trade_price = list(filter(lambda x: x!=0, trade_price))

        # unstable before 400
        plt.plot(trade_price[400:])
        plt.plot(midprice_list[400:])
        #plt.plot(midprice_list)

        plt.legend(['Trade price', 'Mid price'])
        #plt.show()
        plt.savefig('price_'+dt.strftime('%Y_%m_%d')+'.png')

# labeling mid price trend
def labeling():
    client = mg.MongoClient('your mongodb client')
    db = client['twse']
    col = db['tw50']
    code = '2330'

    start_date = datetime(2019, 1, 1)
    end_date = datetime(2019, 1, 10)

    #k=50 or 100 is good
    pred_k = 100

    #threshold=0.0010 is good
    label_threshold = 0.0010

    for dt in date_range(start_date, end_date):
        col = db['tw50']
        docs = col.find( { 'Code': code, 'DspDatetime': { '$gte': dt, '$lt':  dt+timedelta(days=1) } } )

        docs = [doc for doc in docs]

        if docs == []:
            continue


        print(len(docs))

        midprice_list = []

        for doc in docs:
            buypv5 = np.array(doc['BuyPV5']).flatten().tolist()
            sellpv5 = np.array(doc['SellPV5']).flatten().tolist()
            midprice = (buypv5[0]+sellpv5[0])/2
            #print(buypv5+sellpv5, doc['Trade_Price'], midprice)

            midprice_list = midprice_list+[midprice]

        # unstable before 400
        midprice_list = midprice_list[400:]

        df = pd.DataFrame(midprice_list, columns=['Mid price'])


        df['horizon avg'] = 0.0

        #list slicing doesn't include last element; pd.Dataframe loc does include
        for i in df.index:
            df.loc[i,'horizon avg'] = df.loc[i+1:i+pred_k]['Mid price'].sum()/float(pred_k)


        df['pct'] = (df['horizon avg']-df['Mid price'])/df['Mid price']


        #for train_x, train_y
        df['target'] = 0

        #labels 2: equal or greater than 0.00015
        #labels 1: between
        #labels 0: smaller or equal to -0.00015
        df.loc[df['pct'] >=       label_threshold, 'target'] = 1
        df.loc[df['pct'] <=  (-1)*label_threshold, 'target'] = -1

        label = df['target'].values.tolist()
        label = label[:-pred_k]

        #print(label)

        plt.figure(figsize=(40,16), dpi=80)
        plt.rcParams.update({'font.size': 25})

        tans = trans2rect(label)
        tans_stats = sorted(tans, key=lambda x: x[2])

        ax = plt.subplot(111)

        for a in tans:
            if a[0] == 1:
                col = (1,.6,.6)
            elif a[0] == 0:
                col = (1,1,.6)
            elif a[0] == -1:
                col = (.6,1,.6) 

            ax.add_patch(patches.Rectangle((a[1],0), a[2],1, color=col))

        # print(df)
        plt.title('date='+dt.strftime('%Y_%m_%d')+', k={}, threshold={}, #lables={}, max_period={}'.format(pred_k, label_threshold, len(tans), tans_stats[-1][2]))


        midprice_list = df['Mid price'].values.tolist()
        midprice_list = [(float(i)-min(midprice_list))/(max(midprice_list)-min(midprice_list)) for i in midprice_list]
        midprice_list = midprice_list[:-pred_k]

        plt.plot(midprice_list)
        #plt.show()
        plt.savefig(dt.strftime('%Y_%m_%d')+'_k={}_threshold={}.png'.format(pred_k, label_threshold))

if __name__ == '__main__':
    labeling()