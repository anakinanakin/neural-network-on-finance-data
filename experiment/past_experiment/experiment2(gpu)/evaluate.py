import psycopg2, psycopg2.extras
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import patches
from matplotlib.pyplot import figure
from datetime import timedelta, date



def evaluate():
    task = pd.read_csv("task.csv") 

    evaluation = task
    evaluation = evaluation.drop(columns = ['data_set'])

    evaluation['train_loss_min_epoch'] = 0
    evaluation['train_loss_min'] = 100.0
    evaluation['train_loss_100_epoch'] = 100.0
    evaluation['train_loss_200_epoch'] = 100.0
    evaluation['train_loss_300_epoch'] = 100.0
    evaluation['train_loss_400_epoch'] = 100.0
    evaluation['train_loss_500_epoch'] = 100.0

    evaluation['valid_loss_min_epoch'] = 0
    evaluation['valid_loss_min'] = 100.0
    evaluation['valid_loss_100_epoch'] = 100.0
    evaluation['valid_loss_200_epoch'] = 100.0
    evaluation['valid_loss_300_epoch'] = 100.0
    evaluation['valid_loss_400_epoch'] = 100.0
    evaluation['valid_loss_500_epoch'] = 100.0

    evaluation['train_acc_max_epoch'] = 0
    evaluation['train_acc_max'] = 0.0
    evaluation['train_acc_100_epoch'] = 0.0
    evaluation['train_acc_200_epoch'] = 0.0
    evaluation['train_acc_300_epoch'] = 0.0
    evaluation['train_acc_400_epoch'] = 0.0
    evaluation['train_acc_500_epoch'] = 0.0

    evaluation['valid_acc_max_epoch'] = 0
    evaluation['valid_acc_max'] = 0.0
    evaluation['valid_acc_100_epoch'] = 0.0
    evaluation['valid_acc_200_epoch'] = 0.0
    evaluation['valid_acc_300_epoch'] = 0.0
    evaluation['valid_acc_400_epoch'] = 0.0
    evaluation['valid_acc_500_epoch'] = 0.0

    evaluation['train_f1_max_epoch'] = 0
    evaluation['train_f1_max'] = 0.0
    evaluation['train_f1_100_epoch'] = 0.0
    evaluation['train_f1_200_epoch'] = 0.0
    evaluation['train_f1_300_epoch'] = 0.0
    evaluation['train_f1_400_epoch'] = 0.0
    evaluation['train_f1_500_epoch'] = 0.0

    evaluation['valid_f1_max_epoch'] = 0
    evaluation['valid_f1_max'] = 0.0
    evaluation['valid_f1_100_epoch'] = 0.0
    evaluation['valid_f1_200_epoch'] = 0.0
    evaluation['valid_f1_300_epoch'] = 0.0
    evaluation['valid_f1_400_epoch'] = 0.0
    evaluation['valid_f1_500_epoch'] = 0.0

    for index, row in task.iterrows():

        task_id = int(row['task_id'])
        print "taskid:", task_id

        task_dir = os.path.join(os.getcwd(), 'result512_full/'+str(task_id))
        if not os.path.isdir(task_dir):
            continue

        load_dir = os.path.join(task_dir, 'output')
        if not os.path.isdir(load_dir):
            continue

        train_loss = np.load(os.path.join(load_dir, 'train_loss.npy'))
        valid_loss = np.load(os.path.join(load_dir, 'valid_loss.npy'))

        train_acc= np.load(os.path.join(load_dir, 'train_acc.npy'))
        valid_acc= np.load(os.path.join(load_dir, 'valid_acc.npy'))

        train_f1= np.load(os.path.join(load_dir, 'train_f1.npy'))
        valid_f1= np.load(os.path.join(load_dir, 'valid_f1.npy'))

        train_loss_min = np.min(train_loss)
        train_loss_min_epoch = np.argmin(train_loss)+1
        # print "train_loss_min", train_loss_min, "train_loss_min_epoch", train_loss_min_epoch

        valid_loss_min = np.min(valid_loss)
        valid_loss_min_epoch = np.argmin(valid_loss)+1
        # print "valid_loss_min", valid_loss_min, "valid_loss_min_epoch", valid_loss_min_epoch

        train_acc_max = np.max(train_acc)
        train_acc_max_epoch = np.argmax(train_acc)+1
        # print "train_acc_max", train_acc_max, "train_acc_max_epoch", train_acc_max_epoch

        valid_acc_max = np.max(valid_acc)
        valid_acc_max_epoch = np.argmax(valid_acc)+1
        # print "valid_acc_max", valid_acc_max, "valid_acc_max_epoch", valid_acc_max_epoch

        train_f1_max = np.max(train_f1)
        train_f1_max_epoch = np.argmax(train_f1)+1
        # print "train_f1_max", train_f1_max, "train_f1_max_epoch", train_f1_max_epoch

        valid_f1_max = np.max(valid_f1)
        valid_f1_max_epoch = np.argmax(valid_f1)+1
        # print "valid_f1_max", valid_f1_max, "valid_f1_max_epoch", valid_f1_max_epoch

        evaluation['train_loss_min'][index] = train_loss_min
        evaluation['train_loss_min_epoch'][index] = train_loss_min_epoch
        evaluation['train_loss_100_epoch'][index] = train_loss[99]
        evaluation['train_loss_200_epoch'][index] = train_loss[199]
        evaluation['train_loss_300_epoch'][index] = train_loss[299]
        evaluation['train_loss_400_epoch'][index] = train_loss[399]
        evaluation['train_loss_500_epoch'][index] = train_loss[499]

        evaluation['valid_loss_min'][index] = valid_loss_min
        evaluation['valid_loss_min_epoch'][index] = valid_loss_min_epoch
        evaluation['valid_loss_100_epoch'][index] = valid_loss[99]
        evaluation['valid_loss_200_epoch'][index] = valid_loss[199]
        evaluation['valid_loss_300_epoch'][index] = valid_loss[299]
        evaluation['valid_loss_400_epoch'][index] = valid_loss[399]
        evaluation['valid_loss_500_epoch'][index] = valid_loss[499]
        
        evaluation['train_acc_max'][index] = train_acc_max
        evaluation['train_acc_max_epoch'][index] = train_acc_max_epoch
        evaluation['train_acc_100_epoch'][index] = train_acc[99]
        evaluation['train_acc_200_epoch'][index] = train_acc[199]
        evaluation['train_acc_300_epoch'][index] = train_acc[299]
        evaluation['train_acc_400_epoch'][index] = train_acc[399]
        evaluation['train_acc_500_epoch'][index] = train_acc[499]

        evaluation['valid_acc_max'][index] = valid_acc_max
        evaluation['valid_acc_max_epoch'][index] = valid_acc_max_epoch
        evaluation['valid_acc_100_epoch'][index] = valid_acc[99]
        evaluation['valid_acc_200_epoch'][index] = valid_acc[199]
        evaluation['valid_acc_300_epoch'][index] = valid_acc[299]
        evaluation['valid_acc_400_epoch'][index] = valid_acc[399]
        evaluation['valid_acc_500_epoch'][index] = valid_acc[499]

        evaluation['train_f1_max'][index] = train_f1_max
        evaluation['train_f1_max_epoch'][index] = train_f1_max_epoch
        evaluation['train_f1_100_epoch'][index] = train_f1[99]
        evaluation['train_f1_200_epoch'][index] = train_f1[199]
        evaluation['train_f1_300_epoch'][index] = train_f1[299]
        evaluation['train_f1_400_epoch'][index] = train_f1[399]
        evaluation['train_f1_500_epoch'][index] = train_f1[499]

        evaluation['valid_f1_max'][index] = valid_f1_max
        evaluation['valid_f1_max_epoch'][index] = valid_f1_max_epoch
        evaluation['valid_f1_100_epoch'][index] = valid_f1[99]
        evaluation['valid_f1_200_epoch'][index] = valid_f1[199]
        evaluation['valid_f1_300_epoch'][index] = valid_f1[299]
        evaluation['valid_f1_400_epoch'][index] = valid_f1[399]
        evaluation['valid_f1_500_epoch'][index] = valid_f1[499]

    evaluation.to_csv('evaluation.csv')


if __name__ == '__main__':
    evaluate()