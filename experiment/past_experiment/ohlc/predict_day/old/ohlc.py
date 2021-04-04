import psycopg2, psycopg2.extras
import sys
import os
import glob
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from matplotlib import patches
from matplotlib.pyplot import figure
from datetime import timedelta, date
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, precision_score

#error occurs when directly import keras without tensorflow.python
from tensorflow.python.keras import layers, Input, regularizers
from tensorflow.python.keras.backend import clear_session
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import to_categorical, model_to_dot, plot_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, Callback, EarlyStopping




def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)



def get_f1_pre_recall(model, x, y):
    y_pred = model.predict(x, verbose=2)
    y_pred = np.argmax(y_pred, axis=1)
    y_pred.tolist()

    f1 = f1_score(y, y_pred, average='macro')
    precision = precision_score(y, y_pred, average='macro')
    recall = recall_score(y, y_pred, average='macro')

    return f1, precision, recall


def train_model(model, save_dir, task_id, remain_test_x, remain_test_y):

    save_dir_output = os.path.join(save_dir, 'output')
    if not os.path.isdir(save_dir_output):
        os.makedirs(save_dir_output)

    test_acc = []
    test_loss = []
    test_f1 = []
    test_precision = []
    test_recall = []

    remain_test_y_label = [np.where(r==1)[0][0] for r in remain_test_y]


    conn = psycopg2.connect(**eval(open('auth.txt').read()))
    cmd = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    start_date = date(2010, 8, 1)
    end_date = date(2010, 8, 2)

    input_size = 30
    pred_k = 30

    #run from start_date to end_date-1 day
    for single_date in date_range(start_date, end_date):

        cmd.execute('select * from market_index where mid = 1 and dt=%(dt)s',dict(dt=single_date.strftime("%Y-%m-%d")))
        recs = cmd.fetchall()

        if recs == []:
            continue;

        print('start date: '+single_date.strftime("%m/%d/%Y"))

        df = pd.DataFrame(recs, columns = recs[0].keys())

        records = len(df.index)

        test_x = remain_test_x[input_size-1:records-pred_k+1]
        test_y = remain_test_y[input_size-1:records-pred_k+1]
        test_y_label = remain_test_y_label[input_size-1:records-pred_k+1]

        print(test_x.shape)
        print(test_x)

        print(test_y.shape)
        print(test_y)

        print(len(test_y_label))
        print(test_y_label)

        remain_test_x = remain_test_x[records:]
        remain_test_y = remain_test_y[records:]
        remain_test_y_label = remain_test_y_label[records:]

        print(remain_test_x.shape)
        print(remain_test_x)

        print(remain_test_y.shape)
        print(remain_test_y)

        print(len(remain_test_y_label))
        print(remain_test_y_label)


        loss, acc = model.evaluate(test_x, test_y, verbose=2)
        test_loss = test_loss + [loss]
        test_acc =  test_acc + [acc]

        f1, precision, recall = get_f1_pre_recall(model, test_x, test_y_label)

        test_f1 = test_f1 + [f1]
        test_precision = test_precision + [precision]
        test_recall = test_recall + [recall]



    # test_acc = np.array(test_acc)
    # test_loss = np.array(test_loss)
    # test_f1 = np.array(test_f1)
    # test_precision = np.array(test_precision)
    # test_recall = np.array(test_recall)


    # np.save(os.path.join(save_dir, 'output/test_acc.npy'), test_acc)

    # np.save(os.path.join(save_dir, 'output/test_loss.npy'), test_loss)

    # np.save(os.path.join(save_dir, 'output/test_f1.npy'), test_f1)

    # np.save(os.path.join(save_dir, 'output/test_precision.npy'), test_precision)

    # np.save(os.path.join(save_dir, 'output/test_recall.npy'), test_recall)



def main():
    task = pd.read_csv("task.csv") 

    for index, row in task.iterrows():
        
        task_id = int(row['task_id'])

        task_dir = os.path.join(os.getcwd(), str(task_id))
        if not os.path.isdir(task_dir):
            sys.exit('no task')

        load_dir = os.path.join(task_dir, 'data_set')
        if not os.path.isdir(load_dir):
            sys.exit('no data set')

        test_x = np.load(os.path.join(load_dir, 'test_x.npy'))
        test_y = np.load(os.path.join(load_dir, 'test_y_onehot.npy'))

        model_dir = os.path.join(task_dir, 'model')
        if not os.path.isdir(model_dir):
            sys.exit('no model')

        model = load_model(os.path.join(model_dir, 'model_epoch_300.h5'))

        train_model(model, save_dir, task_id, test_x, test_y)



if __name__ == '__main__':
    main()
                 




