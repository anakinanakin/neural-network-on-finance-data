import psycopg2, psycopg2.extras
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
from datetime import timedelta, date
from itertools import combinations
from scipy.stats import ttest_ind, linregress




def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def test():
    conn = psycopg2.connect(**eval(open('auth.txt').read()))
    cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    start_date = date(2010, 6, 1)
    end_date = date(2010, 6, 14)

    prices = []

    for single_date in date_range(start_date, end_date):
        cmd.execute('select * from market_index where mid = 3 and dt=%(dt)s',dict(dt=single_date.strftime("%Y-%m-%d")))
        recs = cmd.fetchall()

        if recs == []:
            continue;

        df = pd.DataFrame(recs, columns = recs[0].keys())

        print(df)


def draw_price_diff():
    conn = psycopg2.connect(**eval(open('auth.txt').read()))



    cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # Monday to Friday, 30min
    cmd.execute(""" select dt, max(close), min(close), max(close)-min(close) diff 
                    from market_index 
                    where mid = 1 
                        and extract(year from dt) = 2010
                        and tm between '09:30' and '10:00'
                        and extract(dow from dt) in (1,2,3,4,5)
                    group by dt
                    order by dt""")

    # no Saturday, 10min
    # cmd.execute(""" select dt, max(close), min(close), max(close)-min(close) diff 
    #                 from market_index 
    #                 where mid = 1 
    #                     and extract(year from dt) = 2010
    #                     and tm < '10:00'
    #                     and extract(dow from dt) in (1,2,3,4,5,7)
    #                 group by dt
    #                 order by dt""")
    

    recs = cmd.fetchall()
    df = pd.DataFrame(recs, columns = recs[0].keys())


    

    cmd2 = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # Monday to Friday, all day
    cmd2.execute(""" select dt, max(close), min(close), max(close)-min(close) diff 
                        from market_index 
                        where mid = 1 
                            and extract(year from dt) = 2010
                            and extract(dow from dt) in (1,2,3,4,5)
                        group by dt
                        order by dt""")

    # # no Saturday
    # # cmd2.execute(""" select dt, max(close), min(close), max(close)-min(close) diff 
    # #                     from market_index 
    # #                     where mid = 1 
    # #                         and extract(year from dt) = 2010
    # #                         and extract(dow from dt) in (1,2,3,4,5,7)
    # #                     group by dt
    # #                     order by dt""")


    recs2 = cmd2.fetchall()
    df2 = pd.DataFrame(recs2, columns = recs2[0].keys())


    cmd3 = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # Monday to Friday, 10hrs
    cmd3.execute(""" select dt, max(close), min(close), max(close)-min(close) diff 
                    from market_index 
                    where mid = 1 
                        and extract(year from dt) = 2010
                        and tm < '10:00'
                        and extract(dow from dt) in (1,2,3,4,5)
                    group by dt
                    order by dt""")

    recs3 = cmd3.fetchall()
    df3 = pd.DataFrame(recs3, columns = recs3[0].keys())


    cmd4 = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    # Monday to Friday, 30min
    cmd4.execute(""" select dt, max(close), min(close), max(close)-min(close) diff 
                    from market_index 
                    where mid = 1 
                        and extract(year from dt) = 2010
                        and tm between '09:30' and '10:30'
                        and extract(dow from dt) in (1,2,3,4,5)
                    group by dt
                    order by dt""")

    recs4 = cmd4.fetchall()
    df4 = pd.DataFrame(recs4, columns = recs4[0].keys())

    # print(df)
    # print(df2)
    # print(df3)
    

    # extra date that doesn't have 9:30~10:00
    df2 = df2[df2['dt'] != date(2010,4,2)]
    df3 = df3[df3['dt'] != date(2010,4,2)]

    diff1 = df['diff'].tolist()
    date1 = df['dt'].tolist()

    diff2 = df2['diff'].tolist()
    date2 = df2['dt'].tolist()

    diff3 = df3['diff'].tolist()
    date3 = df3['dt'].tolist()

    diff4 = df4['diff'].tolist()
    date4 = df4['dt'].tolist()


    data = {'30mins': diff1,'allday': diff2,'10hrs': diff3, '1hr': diff4}


    # for list1, list2 in combinations(data.keys(), 2):
    #     t, p = ttest_ind(data[list1], data[list2])
    #     print list1, list2, p


    #print(df2)
    # df2 = df2.loc[df2['dt'] == date(2010,4,2)]

    # find difference date -> 2010/4/2
    # redundant_date = [i for i in date1 + date2 if i not in date1 or i not in date2] 
    # print(len(date1))
    # print(len(date2))
    # print(redundant_date)

    # m, b = np.polyfit(diff1, diff2, 1)
    # poly1 = m*diff1 + b


    figure(figsize=(32,18), dpi=80)
    plt.rcParams.update({'font.size': 15})

    plt.plot(diff1)
    plt.plot(diff2)
    plt.plot(diff3)
    plt.plot(diff4)
    plt.xlabel('Day')
    plt.ylabel('Max price difference')
    plt.legend(['9:30~10:00', 'one day', '00:00~10:00', '9:30~10:30'])
    plt.savefig('price_diff.png')
    plt.clf()

    figure(figsize=(18,24), dpi=80)

    x_list = [i for i in range(14)]

    # least-squares regression
    slope, intercept, r_value, p_value, std_err = linregress(diff1, diff2)
    y_list = [i*slope+intercept for i in x_list]

    t, p = ttest_ind(diff1, diff2)

    plt.title("""
                9:30~10:00, 
                slope={:0.2e},
                t-statistic={:0.2e}, 
                p-value(null hypothesis:2 samples have identical average values)={:0.2e}, 
                correlation coefficient={:0.2e}, 
                p-value(null hypothesis:slope is zero)={:0.2e}, 
                standard error={:0.2e}""".format(slope, t, p, r_value, p_value, std_err))
    plt.scatter(diff1,diff2)
    plt.plot(y_list)
    plt.legend(['regression'])
    plt.xlabel('30min difference')
    plt.ylabel('One day difference')
    plt.savefig('scatter_30mins.png')
    plt.clf()

    x_list = [i for i in range(45)]

    slope, intercept, r_value, p_value, std_err = linregress(diff3, diff2)
    y_list = [i*slope+intercept for i in x_list]

    t, p = ttest_ind(diff3, diff2)

    plt.title("""
                00:00~10:00, 
                slope={:0.2e},
                t-statistic={:0.2e}, 
                p-value(null hypothesis:2 samples have identical average values)={:0.2e}, 
                correlation coefficient={:0.2e}, 
                p-value(null hypothesis:slope is zero)={:0.2e}, 
                standard error={:0.2e}""".format(slope, t, p, r_value, p_value, std_err))
    plt.scatter(diff3,diff2)
    plt.plot(y_list)
    plt.legend(['regression'])
    plt.xlabel('10hrs difference')
    plt.ylabel('One day difference')
    plt.savefig('scatter_10hrs.png')
    plt.clf()

    x_list = [i for i in range(27)]

    slope, intercept, r_value, p_value, std_err = linregress(diff4, diff2)
    y_list = [i*slope+intercept for i in x_list]

    t, p = ttest_ind(diff4, diff2)

    plt.title("""
                9:30~10:30, 
                slope={:0.2e},
                t-statistic={:0.2e}, 
                p-value(null hypothesis:2 samples have identical average values)={:0.2e}, 
                correlation coefficient={:0.2e}, 
                p-value(null hypothesis:slope is zero)={:0.2e}, 
                standard error={:0.2e}""".format(slope, t, p, r_value, p_value, std_err))
    plt.scatter(diff4,diff2)
    plt.plot(y_list)
    plt.legend(['regression'])
    plt.xlabel('1hr difference')
    plt.ylabel('One day difference')
    plt.savefig('scatter_1hr.png')
    plt.clf()


def output_date():
    conn = psycopg2.connect(**eval(open('auth.txt').read()))
    cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    
    #cmd.execute(""" select dt, array_agg(close), max(close), min(close), max(close)-min(close) diff

    # 10min
    # no Sat,Sun because no ten min data
    cmd.execute(""" select dt, max(close), min(close), max(close)-min(close) diff
                    from market_index 
                    where mid = 1 
                        and extract(year from dt) = 2010
                        and tm < '10:00'
                    group by dt
                    order by dt""")

    recs = cmd.fetchall()

    df = pd.DataFrame(recs, columns = recs[0].keys())

    #print(df)

    #10min close price min&max difference of each day
    diff = df['diff'].tolist()
    date = df['dt'].tolist()

    # print(len(diff))
    # print(diff)
    # print(len(date))
    # print(date)

    threshold = 10

    with open('dt_threshold=10.txt', 'w') as f:
        for i, dt in zip(diff, date):
            if i > threshold:
                f.write(dt.strftime("%Y-%m-%d")+'\n')

if __name__ == '__main__':
    draw_price_diff()
                    
