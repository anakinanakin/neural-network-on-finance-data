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
from math import ceil, sqrt



def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


conn = psycopg2.connect(**eval(open('auth.txt').read()))
cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

start_date = date(2010, 1, 1)
end_date = date(2010, 7, 1)


# set sigma to constant
sig_const = 0


for single_date in date_range(start_date, end_date):
    #smp no volume
    #cmd.execute('select * from market_index where mid = 3 and dt=%(dt)s',dict(dt=single_date.strftime("%Y-%m-%d")))

    #smp with volume
    cmd.execute('select * from market_index where mid = 1 and dt=%(dt)s',dict(dt=single_date.strftime("%Y-%m-%d")))
    recs = cmd.fetchall()

    if recs == []:
        continue;

    df = pd.DataFrame(recs, columns = recs[0].keys())


    df.to_csv(single_date.strftime("%Y-%m-%d")+'.csv')

