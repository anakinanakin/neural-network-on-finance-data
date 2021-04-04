import psycopg2, psycopg2.extras
import sys
import os
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import patches
from matplotlib.pyplot import figure
from datetime import timedelta, date




def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)



task = pd.read_csv("task.csv") 

conn = psycopg2.connect(**eval(open('auth.txt').read()))
cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

start_date = date(2010, 8, 1)
end_date = date(2010, 9, 1)

for single_date in date_range(start_date, end_date):
    print(single_date)

# for index, row in task.iterrows():


#     data_set = int(task['data_set'][index])

#     load_dir = os.path.join(os.getcwd(), 'data_set/'+str(data_set))
#     if not os.path.isdir(load_dir):
#         continue

#     task_id = int(task['task_id'][index])

#     pred_k = int(task['k'][index])

#     label_threshold = float(task['label_threshold'][index])

#     train_y = np.load(os.path.join(load_dir, 'train_y_onehot.npy'))
#     valid_y = np.load(os.path.join(load_dir, 'valid_y_onehot.npy'))
#     test_y = np.load(os.path.join(load_dir, 'test_y_onehot.npy'))

#     train_y = [np.where(r==1)[0][0] for r in train_y]
#     valid_y = [np.where(r==1)[0][0] for r in valid_y]
#     test_y = [np.where(r==1)[0][0] for r in test_y]


#     print('Running experiment {}'.format(task_id))

#     # 0:init
#     # 1:up
#     # 2:down
#     state = 0

    #for i in test_y: 





   






