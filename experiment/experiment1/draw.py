import psycopg2, psycopg2.extras
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import patches
from matplotlib.pyplot import figure
from datetime import timedelta, date





task = pd.read_csv("task.csv") 


def draw():

	for index, row in task.iterrows():

		task_id = int(row['task_id'])

		task_dir = os.path.join(os.getcwd(), 'result/'+str(task_id))
		if not os.path.isdir(task_dir):
			continue

		image_dir = os.path.join(task_dir, 'image')
		if not os.path.isdir(image_dir):
			os.makedirs(image_dir)

		load_dir = os.path.join(task_dir, 'output')
		if not os.path.isdir(load_dir):
			continue

		train_loss = np.load(os.path.join(load_dir, 'train_loss.npy'))
		train_acc = np.load(os.path.join(load_dir, 'train_acc.npy'))
		train_f1= np.load(os.path.join(load_dir, 'train_f1.npy'))
		train_precision = np.load(os.path.join(load_dir, 'train_precision.npy'))
		train_recall = np.load(os.path.join(load_dir, 'train_recall.npy'))

		valid_loss = np.load(os.path.join(load_dir, 'valid_loss.npy'))
		valid_acc = np.load(os.path.join(load_dir, 'valid_acc.npy'))
		valid_f1= np.load(os.path.join(load_dir, 'valid_f1.npy'))
		valid_precision = np.load(os.path.join(load_dir, 'valid_precision.npy'))
		valid_recall = np.load(os.path.join(load_dir, 'valid_recall.npy'))


		#plot train and validation loss
		plt.plot(train_loss)
		plt.plot(valid_loss)
		plt.title('Loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Valid'])
		plt.savefig(os.path.join(image_dir, 'loss.png'))
		plt.clf()


		#plot train and validation accuracy
		plt.plot(train_acc)
		plt.plot(valid_acc)
		plt.title('Accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Valid'])
		plt.savefig(os.path.join(image_dir, 'accuracy.png'))
		plt.clf()

		#plot train and validation accuracy
		plt.plot(train_f1)
		plt.plot(valid_f1)
		plt.title('F1')
		plt.ylabel('F1')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Valid'])
		plt.savefig(os.path.join(image_dir, 'f1.png'))
		plt.clf()

		#plot train and validation accuracy
		plt.plot(train_precision)
		plt.plot(valid_precision)
		plt.title('Precision')
		plt.ylabel('Precision')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Valid'])
		plt.savefig(os.path.join(image_dir, 'precision.png'))
		plt.clf()

		#plot train and validation accuracy
		plt.plot(train_recall)
		plt.plot(valid_recall)
		plt.title('Recall')
		plt.ylabel('Recall')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Valid'])
		plt.savefig(os.path.join(image_dir, 'recall.png'))
		plt.clf()

		plt.plot(train_acc)
		plt.plot(train_recall)
		plt.plot(train_precision)
		plt.plot(train_f1)
		plt.title('Train evaluation')
		plt.ylabel('Score')
		plt.xlabel('Epoch')
		plt.legend(['Accuracy', 'Recall', 'Precision', 'F1'])
		plt.savefig(os.path.join(image_dir, 'train_evaluation.png'))
		plt.clf()

		plt.plot(valid_acc)
		plt.plot(valid_recall)
		plt.plot(valid_precision)
		plt.plot(valid_f1)
		plt.title('Valid evaluation')
		plt.ylabel('Score')
		plt.xlabel('Epoch')
		plt.legend(['Accuracy', 'Recall', 'Precision', 'F1'])
		plt.savefig(os.path.join(image_dir, 'valid_evaluation.png'))
		plt.clf()

def get_loss():

	for index, row in task.iterrows():

		task_id = int(row['task_id'])

		task_dir = os.path.join(os.getcwd(), 'result/'+str(task_id))
		if not os.path.isdir(task_dir):
			continue

		load_dir = os.path.join(task_dir, 'output')
		if not os.path.isdir(load_dir):
			continue

		train_f1= np.load(os.path.join(load_dir, 'train_f1.npy'))
		valid_f1= np.load(os.path.join(load_dir, 'valid_f1.npy'))

		train_loss = np.load(os.path.join(load_dir, 'train_loss.npy'))
		valid_loss = np.load(os.path.join(load_dir, 'valid_loss.npy'))

		train_loss_min = np.min(train_loss)
		valid_loss_min = np.min(valid_loss)

		train_f1_max = np.max(train_f1)
		valid_f1_max = np.max(valid_f1)

		# 8047 valid loss min = 0.599
		#print('task id: '+str(task_id)+'  train loss min: '+str(train_loss_min)+'  valid loss min: '+str(valid_loss_min))

		# 8056 valid f1 max = 0.594
		print('task id: '+str(task_id)+'  train f1 max: '+str(train_f1_max)+'  valid f1 max: '+str(valid_f1_max))


if __name__ == '__main__':
    get_loss()