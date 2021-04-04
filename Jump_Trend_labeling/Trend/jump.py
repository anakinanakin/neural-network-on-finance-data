#source code: https://github.com/alvarobartt/trendet


import psycopg2, psycopg2.extras
import os
import glob
import csv
import time
import datetime
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import patches
from matplotlib.pyplot import figure
from datetime import timedelta, date
from math import ceil, sqrt
from statistics import mean
from unidecode import unidecode



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


def identify_df_trends(df, column, window_size=5, identify='both'):
    """
    This function receives as input a pandas.DataFrame from which data is going to be analysed in order to
    detect/identify trends over a certain date range. A trend is considered so based on the window_size, which
    specifies the number of consecutive days which lead the algorithm to identify the market behaviour as a trend. So
    on, this function will identify both up and down trends and will remove the ones that overlap, keeping just the
    longer trend and discarding the nested trend.
    Args:
        df (:obj:`pandas.DataFrame`): dataframe containing the data to be analysed.
        column (:obj:`str`): name of the column from where trends are going to be identified.
        window_size (:obj:`window`, optional): number of days from where market behaviour is considered a trend.
        identify (:obj:`str`, optional):
            which trends does the user wants to be identified, it can either be 'both', 'up' or 'down'.
    Returns:
        :obj:`pandas.DataFrame`:
            The function returns a :obj:`pandas.DataFrame` which contains the retrieved historical data from Investing
            using `investpy`, with a new column which identifies every trend found on the market between two dates
            identifying when did the trend started and when did it end. So the additional column contains labeled date
            ranges, representing both bullish (up) and bearish (down) trends.
    Raises:
        ValueError: raised if any of the introduced arguments errored.
    """

    if df is None:
        raise ValueError("df argument is mandatory and needs to be a `pandas.DataFrame`.")

    if not isinstance(df, pd.DataFrame):
        raise ValueError("df argument is mandatory and needs to be a `pandas.DataFrame`.")

    if column is None:
        raise ValueError("column parameter is mandatory and must be a valid column name.")

    if column and not isinstance(column, str):
        raise ValueError("column argument needs to be a `str`.")

    if isinstance(df, pd.DataFrame):
        if column not in df.columns:
            raise ValueError("introduced column does not match any column from the specified `pandas.DataFrame`.")
        else:
            if df[column].dtype not in ['int64', 'float64']:
                raise ValueError("supported values are just `int` or `float`, and the specified column of the "
                                 "introduced `pandas.DataFrame` is " + str(df[column].dtype))

    if not isinstance(window_size, int):
        raise ValueError('window_size must be an `int`')

    if isinstance(window_size, int) and window_size < 3:
        raise ValueError('window_size must be an `int` equal or higher than 3!')

    if not isinstance(identify, str):
        raise ValueError('identify should be a `str` contained in [both, up, down]!')

    if isinstance(identify, str) and identify not in ['both', 'up', 'down']:
        raise ValueError('identify should be a `str` contained in [both, up, down]!')

    objs = list()

    up_trend = {
        'name': 'Up Trend',
        'element': np.negative(df['close'])
    }

    down_trend = {
        'name': 'Down Trend',
        'element': df['close']
    }

    if identify == 'both':
        objs.append(up_trend)
        objs.append(down_trend)
    elif identify == 'up':
        objs.append(up_trend)
    elif identify == 'down':
        objs.append(down_trend)

    #print(objs)

    results = dict()

    for obj in objs:
        mov_avg = None
        values = list()

        trends = list()

        for index, value in enumerate(obj['element'], 0):
            # print(index)
            # print(value)

            if mov_avg and mov_avg > value:
                values.append(value)
                mov_avg = mean(values)
            elif mov_avg and mov_avg < value:
                if len(values) > window_size:
                    min_value = min(values)

                    for counter, item in enumerate(values, 0):
                        if item == min_value:
                            break

                    to_trend = from_trend + counter

                    trend = {
                        'from': df.index.tolist()[from_trend],
                        'to': df.index.tolist()[to_trend],
                    }

                    trends.append(trend)

                mov_avg = None
                values = list()
            else:
                from_trend = index

                values.append(value)
                mov_avg = mean(values)

        results[obj['name']] = trends

        # print(results)
        # print("\n\n")


    # deal with overlapping labels, keep longer trends
    if identify == 'both':
        up_trends = list()

        for up in results['Up Trend']:
            flag = True

            for down in results['Down Trend']:
                if (down['from'] <= up['from'] <= down['to']) or (down['from'] <= up['to'] <= down['to']):
                    #print("up")

                    if (up['to'] - up['from']) <= (down['to'] - down['from']):
                        #print("up")
                        flag = False

            for other_up in results['Up Trend']:
                if (other_up['from'] < up['from'] < other_up['to']) or (other_up['from'] < up['to'] < other_up['to']):
                    #print("up")

                    if (up['to'] - up['from']) < (other_up['to'] - other_up['from']):
                        #print("up")
                        flag = False


            if flag is True:
                up_trends.append(up)

        labels = [letter for letter in string.printable[:len(up_trends)]]

        for up_trend, label in zip(up_trends, labels):
            for index, row in df[up_trend['from']:up_trend['to']].iterrows():
                df.loc[index, 'Up Trend'] = label

        down_trends = list()

        for down in results['Down Trend']:
            flag = True

            for up in results['Up Trend']:
                if (up['from'] <= down['from'] <= up['to']) or (up['from'] <= down['to'] <= up['to']):
                    #print("down")

                    if (up['to'] - up['from']) >= (down['to'] - down['from']):
                        #print("down")
                        flag = False


            for other_down in results['Down Trend']:
                if (other_down['from'] < down['from'] < other_down['to']) or (other_down['from'] < down['to'] < other_down['to']):
                    #print("down")

                    if (other_down['to'] - other_down['from']) > (down['to'] - down['from']):
                        #print("down")
                        flag = False


            if flag is True:
                down_trends.append(down)

        labels = [letter for letter in string.printable[:len(down_trends)]]

        for down_trend, label in zip(down_trends, labels):
            for index, row in df[down_trend['from']:down_trend['to']].iterrows():
                df.loc[index, 'Down Trend'] = label

        return df
    elif identify == 'up':
        up_trends = results['Up Trend']

        up_labels = [letter for letter in string.printable[:len(up_trends)]]

        for up_trend, up_label in zip(up_trends, up_labels):
            for index, row in df[up_trend['from']:up_trend['to']].iterrows():
                df.loc[index, 'Up Trend'] = up_label

        return df
    elif identify == 'down':
        down_trends = results['Down Trend']

        down_labels = [letter for letter in string.printable[:len(down_trends)]]

        for down_trend, down_label in zip(down_trends, down_labels):
            for index, row in df[down_trend['from']:down_trend['to']].iterrows():
                df.loc[index, 'Down Trend'] = down_label

        return df





conn = psycopg2.connect(**eval(open('auth.txt').read()))
cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

start_date = date(2010, 3, 25)
end_date = date(2010, 3, 26)

# sampling window
window_size = 5



for single_date in date_range(start_date, end_date):
    #smp no volume
    #cmd.execute('select * from market_index where mid = 3 and dt=%(dt)s',dict(dt=single_date.strftime("%Y-%m-%d")))

    #smp with volume
    cmd.execute('select * from market_index where mid = 1 and dt=%(dt)s',dict(dt=single_date.strftime("%Y-%m-%d")))
    recs = cmd.fetchall()

    if recs == []:
        continue;

    df = pd.DataFrame(recs, columns = recs[0].keys())


    df.sort_values(by='dt')

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df) 

    close_price = df['close'].values
    

    maxprice = max(close_price)
    minprice = min(close_price)

    # prevent from equal to 0
    df['close'] = (df['close']-minprice)/(maxprice - minprice)+0.01

    close_price = df['close'].values

    # close_price = close_price.tolist()

    # df_trend = df.copy()

    # df_trend['Up Trend'] = np.nan
    # df_trend['Down Trend'] = np.nan

    df_trend = identify_df_trends(df, 'close', window_size=window_size, identify='both')

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df_trend) 

    df.reset_index(inplace=True)

    figure(num=None, figsize=(48, 10), dpi=180, facecolor='w', edgecolor='k')

    ax = sns.lineplot(x=df.index, y=df['close'])
    ax.set(xlabel='minute')


    a=0
    b=0

    try:
        labels = df_trend['Up Trend'].dropna().unique().tolist()
    except:
        df_trend['Up Trend'] = np.nan
        a=1

    if a == 0:
        for label in labels:
            ax.axvspan(df[df['Up Trend'] == label].index[0], df[df['Up Trend'] == label].index[-1], alpha=0.2, color='red')



    try:
        labels = df_trend['Down Trend'].dropna().unique().tolist()
    except:
        df_trend['Down Trend'] = np.nan
        b=1

    if b == 0:
        for label in labels:
            ax.axvspan(df[df['Down Trend'] == label].index[0], df[df['Down Trend'] == label].index[-1], alpha=0.2, color='green')
                   

    plt.savefig('date='+single_date.strftime("%m-%d-%Y")+'_window={}.png'.format(window_size))


