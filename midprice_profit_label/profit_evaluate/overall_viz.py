import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
from statistics import mean, median


total_return = np.load('total_return.npy')
trade_time = np.load('trade_time.npy')

def sharpe_ratio(total_return, trade_time, data_type):
	for k in range(20):
		if data_type == 1:
			data_list = total_return[:,k,:]-trade_time[:,k,:]*1
		elif data_type == 2:
			data_list = total_return[:,k,:]-trade_time[:,k,:]*2
		elif data_type == 3:
			data_list = total_return[:,k,:]-trade_time[:,k,:]*3
		else:
			sys.exit("type incorrect")

	print(len(data_list))
	print(data_list)

def draw_k(total_return, trade_time, data_type):
	total = []
	avg_list = [np.nan]
	med_list = [np.nan]
	zero_list = [np.nan]

	for k in range(20):
		if data_type == 1:
			data_list = trade_time[:,k,:]
		elif data_type == 2:
			data_list = total_return[:,k,:]
		elif data_type == 3:
			data_list = total_return[:,k,:]/trade_time[:,k,:]
		elif data_type == 4:
			data_list = total_return[:,k,:]-trade_time[:,k,:]*1
		elif data_type == 5:
			data_list = total_return[:,k,:]-trade_time[:,k,:]*2
		elif data_type == 6:
			data_list = total_return[:,k,:]-trade_time[:,k,:]*3
		else:
			sys.exit("type incorrect")


		trade_time_month = []

		for column in range(len(data_list[0,:])):
			trade_time_month = trade_time_month + [sum(row[column] for row in data_list)]

		# print(len(trade_time_month))
		#print(trade_time_month)

		avg = mean(trade_time_month)
		

		med = median(trade_time_month)


		total = total + [trade_time_month]

		avg_list = avg_list + [avg]

		med_list = med_list + [med]

		zero_list = zero_list + [0]

	return total, avg_list, med_list, zero_list

def draw_threshold(total_return, trade_time, data_type):
	total = []
	avg_list = [np.nan]
	med_list = [np.nan]
	zero_list = [np.nan]

	for th in range(24):
		if data_type == 1:
			data_list = trade_time[:,:,th]
		elif data_type == 2:
			data_list = total_return[:,:,th]
		elif data_type == 3:
			data_list = total_return[:,:,th]/trade_time[:,:,th]
		elif data_type == 4:
			data_list = total_return[:,:,th]-trade_time[:,:,th]*1
		elif data_type == 5:
			data_list = total_return[:,:,th]-trade_time[:,:,th]*2
		elif data_type == 6:
			data_list = total_return[:,:,th]-trade_time[:,:,th]*3
		else:
			sys.exit("type incorrect")


		trade_time_month = []

		for column in range(len(data_list[0,:])):
			trade_time_month = trade_time_month + [sum(row[column] for row in data_list)]

		# print(len(trade_time_month))
		#print(trade_time_month)

		avg = mean(trade_time_month)


		med = median(trade_time_month)


		total = total + [trade_time_month]

		avg_list = avg_list + [avg]

		med_list = med_list + [med]

		zero_list = zero_list + [0]

	return total, avg_list, med_list, zero_list

def draw_day(total_return, trade_time, data_type):
	total = []
	avg_list = [np.nan]
	med_list = [np.nan]
	zero_list = [np.nan]


	with open('dt_all.txt', 'r') as f:
		dates = f.readlines()

		for index, date in enumerate(dates):
			if data_type == 1:
				data_list = trade_time[index,:,:]
			elif data_type == 2:
				data_list = total_return[index,:,:]
			elif data_type == 3:
				data_list = total_return[index,:,:]/trade_time[index,:,:]
			elif data_type == 4:
				data_list = total_return[index,:,:]-trade_time[index,:,:]*1
			elif data_type == 5:
				data_list = total_return[index,:,:]-trade_time[index,:,:]*2
			elif data_type == 6:
				data_list = total_return[index,:,:]-trade_time[index,:,:]*3
			else:
				sys.exit("type incorrect")


			trade_time_month = list()

			# print(len(data_list))
			# print(len(data_list[0,:]))
			# print(data_list)

			for l in data_list:
				# trade_time_month = trade_time_month + [sum(row[column] for row in data_list)]

				#plot every point, no aggregation
				trade_time_month.extend(l)

			#print(trade_time_month)

			# print(len(trade_time_month))
			#print(trade_time_month)

			avg = mean(trade_time_month)

			med = median(trade_time_month)


			total = total + [trade_time_month]

			avg_list = avg_list + [avg]

			med_list = med_list + [med]

			zero_list = zero_list + [0]

	return total, avg_list, med_list, zero_list

def plot_by_k():
	plt.rcParams.update({'font.size': 20})
	figure(figsize=(80,32), dpi=80)

	plt.suptitle('2010/06 K')

	total_trade_time, avg_list , med_list, zero_list = draw_k(total_return, trade_time, 1)

	plt.subplot(231)
	plt.title('#Trades')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_trade_time)
	plt.xticks(np.arange(1, 21, step=1), (str((i+1)*10) for i in range(20)))
	plt.xlabel('K')
	plt.ylabel('#Trades')
	#plt.show()
	# plt.savefig('#Trades.png')
	# plt.clf()



	total_trade_return, avg_list , med_list, zero_list = draw_k(total_return, trade_time, 2)

	plt.subplot(232)
	plt.title('Total Profit')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_trade_return)
	plt.xticks(np.arange(1, 21, step=1), (str((i+1)*10) for i in range(20)))
	plt.xlabel('K')
	plt.ylabel('Profit')
	#plt.show()
	# plt.savefig('total_profit.png')
	# plt.clf()


	total_avg_return, avg_list , med_list, zero_list = draw_k(total_return, trade_time, 3)

	plt.subplot(233)
	plt.title('Average Profit')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_avg_return)
	plt.xticks(np.arange(1, 21, step=1), (str((i+1)*10) for i in range(20)))
	plt.xlabel('K')
	plt.ylabel('Profit')
	#plt.show()
	# plt.savefig('average_profit.png')
	# plt.clf()



	total_cost_return1, avg_list , med_list, zero_list = draw_k(total_return, trade_time, 4)

	plt.subplot(234)
	plt.title('Profit with cost bp1')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_cost_return1)
	plt.xticks(np.arange(1, 21, step=1), (str((i+1)*10) for i in range(20)))
	plt.xlabel('K')
	plt.ylabel('Profit')
	#plt.show()
	# plt.savefig('profit_cost1.png')
	# plt.clf()

	total_cost_return2, avg_list , med_list, zero_list = draw_k(total_return, trade_time, 5)

	plt.subplot(235)
	plt.title('Profit with cost bp2')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_cost_return2)
	plt.xticks(np.arange(1, 21, step=1), (str((i+1)*10) for i in range(20)))
	plt.xlabel('K')
	plt.ylabel('Profit')
	#plt.show()
	# plt.savefig('profit_cost2.png')
	# plt.clf()

	total_cost_return3, avg_list , med_list, zero_list = draw_k(total_return, trade_time, 6)

	plt.subplot(236)
	plt.title('Profit with cost bp3')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_cost_return3)
	plt.xticks(np.arange(1, 21, step=1), (str((i+1)*10) for i in range(20)))
	plt.xlabel('K')
	plt.ylabel('Profit')
	#plt.show()
	plt.savefig('201006_k.png')
	plt.clf()

def plot_by_threshold():

	plt.rcParams.update({'font.size': 20})
	figure(figsize=(80,32), dpi=80)

	plt.suptitle('2010/06 Threshold')
	total_trade_time, avg_list , med_list, zero_list = draw_threshold(total_return, trade_time, 1)


	plt.subplot(231)
	plt.title('#Trades')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_trade_time)
	plt.xticks(np.arange(1, 25, step=1), (str(i+2) for i in range(24)))
	plt.xlabel('Threshold')
	plt.ylabel('#Trades')
	#plt.show()
	# plt.savefig('#Trades.png')
	# plt.clf()



	total_trade_return, avg_list , med_list, zero_list = draw_threshold(total_return, trade_time, 2)

	plt.subplot(232)
	plt.title('Total Profit')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_trade_return)
	plt.xticks(np.arange(1, 25, step=1), (str(i+2) for i in range(24)))
	plt.xlabel('Threshold')
	plt.ylabel('Profit')
	#plt.show()
	# plt.savefig('total_profit.png')
	# plt.clf()


	total_avg_return, avg_list , med_list, zero_list = draw_threshold(total_return, trade_time, 3)

	plt.subplot(233)
	plt.title('Average Profit')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_avg_return)
	plt.xticks(np.arange(1, 25, step=1), (str(i+2) for i in range(24)))
	plt.xlabel('Threshold')
	plt.ylabel('Profit')
	#plt.show()
	# plt.savefig('average_profit.png')
	# plt.clf()



	total_cost_return1, avg_list , med_list, zero_list = draw_threshold(total_return, trade_time, 4)

	plt.subplot(234)
	plt.title('Profit with cost bp1')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_cost_return1)
	plt.xticks(np.arange(1, 25, step=1), (str(i+2) for i in range(24)))
	plt.xlabel('Threshold')
	plt.ylabel('Profit')
	#plt.show()
	# plt.savefig('profit_cost1.png')
	# plt.clf()

	total_cost_return2, avg_list , med_list, zero_list = draw_threshold(total_return, trade_time, 5)

	plt.subplot(235)
	plt.title('Profit with cost bp2')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_cost_return2)
	plt.xticks(np.arange(1, 25, step=1), (str(i+2) for i in range(24)))
	plt.xlabel('Threshold')
	plt.ylabel('Profit')
	#plt.show()
	# plt.savefig('profit_cost2.png')
	# plt.clf()

	total_cost_return3, avg_list , med_list, zero_list = draw_threshold(total_return, trade_time, 6)

	plt.subplot(236)
	plt.title('Profit with cost bp3')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_cost_return3)
	plt.xticks(np.arange(1, 25, step=1), (str(i+2) for i in range(24)))	
	plt.xlabel('Threshold')
	plt.ylabel('Profit')
	#plt.show()
	plt.savefig('201006_threshold.png')
	plt.clf()

def plot_by_day():
	figure(figsize=(200,80), dpi=80)
	plt.rcParams.update({'font.size': 35})

	plt.suptitle('2010 Price all')
	# plt.subplots_adjust(wspace=0.5)
	# plt.subplots_adjust(hspace=0.5)



	total_trade_time, avg_list, med_list, zero_list = draw_day(total_return, trade_time, 1)

	days = len(avg_list)

	plt.subplot(231)
	plt.title('#Trades')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_trade_time)
	plt.xticks(np.arange(1, 260, step=20), (str(i*20+1) for i in range(days)))
	plt.xlabel('Day')
	plt.ylabel('#Trades')


	total_trade_return, avg_list, med_list, zero_list = draw_day(total_return, trade_time, 2)

	plt.subplot(232)
	plt.title('Total Profit')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_trade_return)
	plt.xticks(np.arange(1, 260, step=20), (str(i*20+1) for i in range(days)))
	plt.xlabel('Day')
	plt.ylabel('Profit')


	total_avg_return, avg_list, med_list, zero_list = draw_day(total_return, trade_time, 3)

	plt.subplot(233)
	plt.title('Average Profit')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_avg_return)
	plt.xticks(np.arange(1, 260, step=20), (str(i*20+1) for i in range(days)))
	plt.xlabel('Day')
	plt.ylabel('Profit')


	total_cost_return1, avg_list, med_list, zero_list = draw_day(total_return, trade_time, 4)

	plt.subplot(234)
	plt.title('Profit with cost bp1')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_cost_return1)
	plt.xticks(np.arange(1, 260, step=20), (str(i*20+1) for i in range(days)))
	plt.xlabel('Day')
	plt.ylabel('Profit')


	total_cost_return2, avg_list, med_list, zero_list = draw_day(total_return, trade_time, 5)

	plt.subplot(235)
	plt.title('Profit with cost bp2')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_cost_return2)
	plt.xticks(np.arange(1, 260, step=20), (str(i*20+1) for i in range(days)))	
	plt.xlabel('Day')
	plt.ylabel('Profit')


	total_cost_return3, avg_list, med_list, zero_list = draw_day(total_return, trade_time, 6)

	plt.subplot(236)
	plt.title('Profit with cost bp3')
	plt.plot(avg_list)
	plt.plot(med_list)
	plt.plot(zero_list)
	plt.legend(['Average', 'Median', 'Zero'])
	plt.boxplot(total_cost_return3)	
	plt.xticks(np.arange(1, 260, step=20), (str(i*20+1) for i in range(days)))
	plt.xlabel('Day')
	plt.ylabel('Profit')


	plt.savefig('price_all.png')

def plot_heatmap():
	plt.rcParams.update({'font.size': 20})
	figure(figsize=(80,32), dpi=80)

	plt.suptitle('2010/01~06')

	total_trade_time, avg_list , med_list, zero_list = draw_k(total_return, trade_time, 1)

	plt.subplot(231)
	plt.title('#Trades')
	plt.imshow(total_trade_time,cmap='YlOrRd')
	plt.colorbar()
	plt.xticks(np.arange(0, 51, step=5), (str((i*5)) for i in range(51)))
	plt.yticks(np.arange(0, 20, step=1), (str((i+1)*10) for i in range(20)))
	plt.xlabel('Threshold')
	plt.ylabel('K')


	total_trade_return, avg_list , med_list, zero_list = draw_k(total_return, trade_time, 2)

	plt.subplot(232)
	plt.title('Total Profit')
	plt.imshow(total_trade_return,cmap='YlOrRd')
	plt.colorbar()
	plt.xticks(np.arange(0, 51, step=5), (str((i*5)) for i in range(51)))
	plt.yticks(np.arange(0, 20, step=1), (str((i+1)*10) for i in range(20)))
	plt.xlabel('Threshold')
	plt.ylabel('K')


	total_avg_return, avg_list , med_list, zero_list = draw_k(total_return, trade_time, 3)

	plt.subplot(233)
	plt.title('Average Profit')
	plt.imshow(total_avg_return,cmap='YlOrRd')
	plt.colorbar()
	plt.xticks(np.arange(0, 51, step=5), (str((i*5)) for i in range(51)))
	plt.yticks(np.arange(0, 20, step=1), (str((i+1)*10) for i in range(20)))
	plt.xlabel('Threshold')
	plt.ylabel('K')



	total_cost_return1, avg_list , med_list, zero_list = draw_k(total_return, trade_time, 4)

	plt.subplot(234)
	plt.title('Profit with cost bp1')
	plt.imshow(total_cost_return1,cmap='tab20c')
	plt.colorbar()
	plt.xticks(np.arange(0, 51, step=5), (str((i*5)) for i in range(51)))
	plt.yticks(np.arange(0, 20, step=1), (str((i+1)*10) for i in range(20)))
	plt.xlabel('Threshold')
	plt.ylabel('K')



	total_cost_return2, avg_list , med_list, zero_list = draw_k(total_return, trade_time, 5)

	plt.subplot(235)
	plt.title('Profit with cost bp2')
	plt.imshow(total_cost_return2,cmap='tab20c')
	plt.colorbar()
	plt.xticks(np.arange(0, 51, step=5), (str((i*5)) for i in range(51)))
	plt.yticks(np.arange(0, 20, step=1), (str((i+1)*10) for i in range(20)))
	plt.xlabel('Threshold')
	plt.ylabel('K')



	total_cost_return3, avg_list , med_list, zero_list = draw_k(total_return, trade_time, 6)

	plt.subplot(236)
	plt.title('Profit with cost bp3')
	plt.imshow(total_cost_return3,cmap='tab20c')
	plt.colorbar()
	plt.xticks(np.arange(0, 51, step=5), (str((i*5)) for i in range(51)))
	plt.yticks(np.arange(0, 20, step=1), (str((i+1)*10) for i in range(20)))
	plt.xlabel('Threshold')
	plt.ylabel('K')



	plt.savefig('201001~06_heatmap.png')
	plt.clf()



if __name__ == '__main__':
	#plot_by_k()
	#plot_by_threshold()
	#plot_by_day()
	#draw_day(total_return, trade_time, 1)
	plot_heatmap()
	#sharpe_ratio(total_return, trade_time, 1)

