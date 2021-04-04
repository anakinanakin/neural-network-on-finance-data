import psycopg2, psycopg2.extras
import time
import numpy as np
import pandas as pd

from datetime import timedelta, date




def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def generate_by_month():
    conn = psycopg2.connect(**eval(open('auth.txt').read()))
    cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    start_date = date(2005, 1, 1)
    end_date = date(2005, 7, 1)

    prices = []

    with open('dt_index.txt', 'w') as f:
        #run from start_date to end_date-1 day
        for single_date in date_range(start_date, end_date):
            cmd.execute('select * from market_index where mid = 1 and dt=%(dt)s',dict(dt=single_date.strftime("%Y-%m-%d")))
            recs = cmd.fetchall()

            if recs == []:
                continue;

            df = pd.DataFrame(recs, columns = recs[0].keys())

            prices = prices + [df['close'].tolist()]

            f.write(single_date.strftime("%Y-%m-%d")+'\n')


    # print(prices)
    # print(np.shape(prices))


    np.save(('price_200501~06.npy'), prices)

def generate_by_txt():
    conn = psycopg2.connect(**eval(open('auth.txt').read()))
    cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    prices = []

    with open('dt_threshold=10.txt', 'r') as f:
        dates = f.readlines()
        # print(dates)

        for index, date in enumerate(dates):
            cmd.execute('select * from market_index where mid = 1 and dt=%(dt)s',dict(dt=date))
            recs = cmd.fetchall()

            if recs == []:
                continue;

            df = pd.DataFrame(recs, columns = recs[0].keys())

            prices = prices + [df['close'].tolist()]

    # print(prices)
    # print(np.shape(prices))


    np.save(('price_threshold=10.npy'), prices)


if __name__ == '__main__':
    generate_by_month()
                    
