# python3 construct.py 0 8 &
# python3 construct.py 1 8 &
# python3 construct.py 2 8 &
# python3 construct.py 3 8 &
# python3 construct.py 4 8 &
# python3 construct.py 5 8 &
# python3 construct.py 6 8 &
# python3 construct.py 7 8 &


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from statistics import mean



def weird_division(n, d):
	return n / d if d else 0

def labeling(price, bp, k):
	label = []
	for i, p, in enumerate(price[:-k]):
		slope = (mean(price[i+1:i+k+1])-price[i])/price[i]
		if abs(slope) < bp:
			label.append(0)
		else:
			label.append(weird_division(slope,abs(slope)))
	return label

def calc_profit(price, label):
	total_profit = 0
	trade_times = 0
	status = label[0]
	start_price = price[0]
	for i, lbl in enumerate(label[1:]):
		if lbl == status or lbl == 0:
			continue
		else:
			total_profit += (price[i+1]-start_price)*status
			# print((price[i+1]-start_price),status)
			trade_times += 1 if status != 0 else 0
			status = lbl
			start_price = price[i+1]

	if trade_times == 0:
		trade_times = 1

	return total_profit, trade_times



def main():
	prices = np.load('price_200501~06.npy', allow_pickle=True)
	thread_no = int(sys.argv[1])
	thread_num = 8
	if len(sys.argv) > 2:
		thread_num =int(sys.argv[2])

	# lbl = labeling(prices[0], 0.0,10)
	# # print(lbl)
	# print(calc_profit(prices[0], lbl))
	# plot_labeling(prices[0], lbl)
	### apply thread channel:
	### usage:
	### python construct.py 0 8

	return_dir = os.path.join(os.getcwd(), 'total_return/')
	if not os.path.isdir(return_dir):
		os.makedirs(return_dir)   

	time_dir = os.path.join(os.getcwd(), 'trade_time/')
	if not os.path.isdir(time_dir):
		os.makedirs(time_dir)   

	for i, price in enumerate(prices):
		if i % thread_num != thread_no:
			continue
		total_profit_mat = []
		trade_time_mat = []
		for k in range(10, 201, 10):
			total_profit_vec = []
			trade_time_vec = []
			for bp in np.arange(0.0000,0.0051,0.0001):
				lbl = labeling(price, bp, k)
				tr, tt = calc_profit(price, lbl)
				# print(tr)
				total_profit_vec.append(tr)
				trade_time_vec.append(tt)
			total_profit_mat.append(total_profit_vec)
			trade_time_mat.append(trade_time_vec)
		np.save('total_return/{}.npy'.format(i),total_profit_mat)
		np.save('trade_time/{}.npy'.format(i),trade_time_mat)

def trans2rect(label):
	status = label[0]
	position = 0
	width = 1
	rects = []
	for i,l in enumerate(label[1:]):
		if status == l:
			width += 1
		else:
			rects.append((status,position,width))
			status = l
			position = i+1
			width = 1
	return rects

def plot_labeling(price, label):
	rects = trans2rect(label)
	ymin, ymax = min(price), max(price)-min(price)
	for rect in rects:
		color = (1,1,1)
		if rect[0]==1: color = (1, .2, .2)
		elif rect[0]==-1: color = (.2, 1, .2)
		elif rect[0]==0: color = (.8, .8, .8)
		plt.gca().add_patch(patches.Rectangle((rect[1], ymin), rect[2], ymax,color=color, alpha=0.5))
	plt.plot(price)
	plt.show()

if __name__ == '__main__':
	main()


