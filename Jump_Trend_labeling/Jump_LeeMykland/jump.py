#source code: https://gist.github.com/linuskohl/690da335a34ebf1cfc5ab27973e16ee5
#other code: https://www.wilsonmongwe.co.za/a-test-for-the-presence-of-jumps-in-financial-markets-using-neural-networks-in-r/
#			 https://www.mathworks.com/matlabcentral/fileexchange/71058-lee-mykland-nonparametric-jump-detection


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


def movmean(v, kb):
    """
    Computes the mean with a window of length kb+kf+1 that includes the element 
    in the current position, kb elements backward, and kf elements forward.
    Nonexisting elements at the edges get substituted with NaN.
    Args:
        v (list(float)): List of values.
        kb (int): Number of elements to include before current position
    Returns:
        list(float): List of the same size as v containing the mean values
    """
    m = len(v) * [np.nan]
    #print(m)

    for i in range(kb, len(v)):
       m[i] = np.mean(v[i-kb:i+1])

    #print(m)
    return m


def LeeMykland(S, sampling, tm, significance_level = 0.01, sig_const = 1):
    """
    "Jumps in Financial Markets: A NewNonparametric Test and Jump Dynamics"
    - by Suzanne S. Lee and Per A. Mykland
    
    "https://galton.uchicago.edu/~mykland/paperlinks/LeeMykland-2535.pdf"
    
    Args:
        S (list(float)): An array containing prices, where each entry 
                         corresponds to the price sampled every 'sampling' minutes.
        sampling (int): Minutes between entries in S
        tm (int): trading minutes
        significance_level (float): Defaults to 1% (0.01)
        
    Returns:
        A pandas dataframe containing a row covering the interval 
        [t_i, t_i+sampling] containing the following values:
        J:   Binary value is jump with direction (sign)
        L:   L statistics
        T:   Test statistics
        sig: Volatility estimate
    """
    np.set_printoptions(threshold=np.inf)

    #tm = 252*24*60

    # a way to estimate an appropriate k(window size)
    k   = int(ceil(sqrt(tm/sampling)))
    #k   = int(ceil(sqrt(tm/sampling)))*5
    #k = 10

    r = np.append(np.nan, np.diff(np.log(S)))
    #print(np.log(S))
    #print(r)

    bpv_front = np.absolute(r[:])
    bpv_back = np.absolute(np.append(np.nan, r[:-1]))

    #print(bpv_front)
    #print(bpv_back)

    bpv = np.multiply(bpv_front, bpv_back)
    #print(bpv)

    #bpv = np.append(np.nan, bpv[0:-1]).reshape(-1,1) # Realized bipower variation
    #print(bpv)

    sig = np.sqrt(movmean(bpv, k-3)) # Volatility estimate

    #print(sig)

    L   = r/sig
    #L = r/sig_const

    n   = np.size(S) # Length of S
    c   = (2/np.pi)**0.5
    Sn  = 1/(c*((2*np.log(n))**0.5))
    Cn  = ((2*np.log(n))**0.5)/c - np.log(np.pi*np.log(n))/(2*c*((2*np.log(n))**0.5))
    beta_star   = -np.log(-np.log(1-significance_level)) # Jump threshold
    T   = (abs(L)-Cn)/Sn
    J   = (T > beta_star).astype(float)
    J   = J*np.sign(r) # Add direction
    # First k rows are NaN involved in bipower variation estimation are set to NaN.
    J[0:k] = np.nan
    # Build and retunr result dataframe
    return pd.DataFrame({'L': L,'sig': sig, 'T': T,'J':J})




conn = psycopg2.connect(**eval(open('auth.txt').read()))
cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

start_date = date(2010, 3, 24)
end_date = date(2010, 3, 25)

# sampling window
delta_t = 1

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


    df.sort_values(by='dt')

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df) 

    close_price = df['close'].values
    

    maxprice = max(close_price)
    minprice = min(close_price)

    df['close'] = (df['close']-minprice)/(maxprice - minprice)+0.01

    close_price = df['close'].values

    close_price = close_price.tolist()

    close_price_new = []

    for i in range(len(close_price)):
        if i%delta_t == 0:
            close_price_new = close_price_new + [close_price[i]]



    # print(close_price)
    # #close_price = close_price + [1163]
    #print(close_price_new)
    # print(len(close_price_new))

    jump_df = LeeMykland(close_price_new, sampling = delta_t, tm = len(close_price_new), significance_level = 0.01, sig_const = sig_const)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(jump_df) 

    jumps = jump_df['J'].values

    jumps_new = []

    for i in range(len(jumps)):
        for j in range(delta_t):
            jumps_new = jumps_new + [jumps[i]]

    # print(jumps_new)
    # print(len(jumps_new))

    L = jump_df['L'].values

    #ignore nan
    maxprice = np.nanmax(L[L != np.inf])
    minprice = np.nanmin(L[L != -np.inf])

    #print(maxprice)
    #print(minprice)

    jump_df['L'] = (jump_df['L']-minprice)/(maxprice - minprice)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(jump_df) 

    L = jump_df['L'].values

    L = L.tolist()

    #print(L)

    figure(num=None, figsize=(48, 10), dpi=180, facecolor='w', edgecolor='k')

    ax = plt.subplot(111)
    tans = trans2rect(jumps_new)

    #tans_stats = sorted(tans, key=lambda x: x[2])

    for a in tans:
        if a[0] == -1:
            col = (.6, 1,.6)
        elif a[0] == 1:
            col = (1,.6,.6)
        elif a[0] == 0:
            col = (1, 1,.6)
        else:
            col = 'w'
        ax.add_patch(patches.Rectangle((a[1],0), a[2],1, color=col))


    plt.plot(close_price)
    plt.plot(L)

    plt.legend(['Close price', 'L'])
    plt.title('sampling#lables: {}'.format(len(tans)))
    #plt.savefig('jump_sig={}.png'.format(sig_const))
    plt.savefig('jump.png')

