import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure



total_return = np.load('total_return.npy')
trade_time = np.load('trade_time.npy')


def plot_by_k():
	figure(figsize=(16,6), dpi=150)

	for k in range(10):
		plt.suptitle('K: {}'.format((k+1)*10))

		plt.subplot(231)
		plt.title('#Trades')
		plt.xlabel('Threshold')
		plt.ylabel('Day')
		plt.xticks(np.arange(0, 51, step=10))
		plt.yticks(np.arange(0, 30, step=6))
		plt.imshow(trade_time[:,k,:],cmap='coolwarm')
		plt.colorbar()


		plt.subplot(232)
		plt.title('Total Profit')
		plt.xlabel('Threshold')
		plt.ylabel('Day')
		plt.xticks(np.arange(0, 51, step=10))
		plt.yticks(np.arange(0, 30, step=6))
		plt.imshow(total_return[:,k,:],cmap='coolwarm')
		plt.colorbar()

		plt.subplot(233)
		plt.title('Average Profit')
		plt.xlabel('Threshold')
		plt.ylabel('Day')
		plt.xticks(np.arange(0, 51, step=10))
		plt.yticks(np.arange(0, 30, step=6))
		plt.imshow(total_return[:,k,:]/trade_time[:,k,:],cmap='coolwarm')
		plt.colorbar()

		plt.subplot(234)
		plt.title('Profit with Cost bp1')
		plt.xlabel('Threshold')
		plt.ylabel('Day')
		plt.xticks(np.arange(0, 51, step=10))
		plt.yticks(np.arange(0, 30, step=6))
		plt.imshow(total_return[:,k,:]-trade_time[:,k,:]*1,cmap='coolwarm')
		plt.colorbar()

		plt.subplot(235)
		plt.title('Profit with Cost bp2')
		plt.xlabel('Threshold')
		plt.ylabel('Day')
		plt.xticks(np.arange(0, 51, step=10))
		plt.yticks(np.arange(0, 30, step=6))
		plt.imshow(total_return[:,k,:]-trade_time[:,k,:]*2,cmap='coolwarm')
		plt.colorbar()

		plt.subplot(236)
		plt.title('Profit with Cost bp3')
		plt.xlabel('Threshold')
		plt.ylabel('Day')
		plt.xticks(np.arange(0, 51, step=10))
		plt.yticks(np.arange(0, 30, step=6))
		plt.imshow(total_return[:,k,:]-trade_time[:,k,:]*3,cmap='coolwarm')
		plt.colorbar()

		#plt.show()
		plt.savefig('k{}.png'.format((k+1)*10))
		plt.clf()



def plot_by_threshold():
	figure(figsize=(16,6), dpi=150)

	for th in range(51):
		plt.suptitle('threshold: {}'.format(th))
		plt.subplots_adjust(hspace=0.5)

		plt.subplot(231)
		plt.title('#Trades')
		plt.xlabel('K')
		plt.ylabel('Day')
		plt.xticks(np.arange(0, 11, step=5), ('10', '50', '100'))
		plt.yticks(np.arange(0, 30, step=6))
		plt.imshow(trade_time[:,:,th],cmap='coolwarm')
		plt.colorbar()

		plt.subplot(232)
		plt.title('Total Profit')
		plt.xlabel('K')
		plt.ylabel('Day')
		plt.xticks(np.arange(0, 11, step=5), ('10', '50', '100'))
		plt.yticks(np.arange(0, 30, step=6))
		plt.imshow(total_return[:,:,th],cmap='coolwarm')
		plt.colorbar()

		plt.subplot(233)
		plt.title('Average Profit')
		plt.xlabel('K')
		plt.ylabel('Day')
		plt.xticks(np.arange(0, 11, step=5), ('10', '50', '100'))
		plt.yticks(np.arange(0, 30, step=6))
		plt.imshow(total_return[:,:,th]/trade_time[:,:,th],cmap='coolwarm')
		plt.colorbar()

		plt.subplot(234)
		plt.title('Profit with Cost bp1')
		plt.xlabel('K')
		plt.ylabel('Day')
		plt.xticks(np.arange(0, 11, step=5), ('10', '50', '100'))
		plt.yticks(np.arange(0, 30, step=6))
		plt.imshow(total_return[:,:,th]-trade_time[:,:,th]*1,cmap='coolwarm')
		plt.colorbar()

		plt.subplot(235)
		plt.title('Profit with Cost bp2')
		plt.xlabel('K')
		plt.ylabel('Day')
		plt.xticks(np.arange(0, 11, step=5), ('10', '50', '100'))
		plt.yticks(np.arange(0, 30, step=6))
		plt.imshow(total_return[:,:,th]-trade_time[:,:,th]*2,cmap='coolwarm')
		plt.colorbar()

		plt.subplot(236)
		plt.title('Profit with Cost bp3')
		plt.xlabel('K')
		plt.ylabel('Day')
		plt.xticks(np.arange(0, 11, step=5), ('10', '50', '100'))
		plt.yticks(np.arange(0, 30, step=6))
		plt.imshow(total_return[:,:,th]-trade_time[:,:,th]*3,cmap='coolwarm')
		plt.colorbar()

		# plt.show()
		plt.savefig('threshold{}.png'.format(th))
		plt.clf()



def plot_by_day():
	figure(figsize=(16,6), dpi=80)

	with open('dt_threshold=20.txt', 'r') as f:
		dates = f.readlines()
		#print(dates)

		for index, date in enumerate(dates):
			#print(date)

			plt.suptitle(date)

			plt.subplot(231)
			plt.title('#Trades')
			plt.xlabel('Threshold')
			plt.ylabel('K')
			plt.xticks(np.arange(0, 51, step=10))
			plt.yticks(np.arange(0, 11, step=5), ('10', '50', '100'))
			plt.imshow(trade_time[index,:,:],cmap='coolwarm')
			plt.colorbar()


			plt.subplot(232)
			plt.title('Total Profit')
			plt.xlabel('Threshold')
			plt.ylabel('K')
			plt.xticks(np.arange(0, 51, step=10))
			plt.yticks(np.arange(0, 11, step=5), ('10', '50', '100'))
			plt.imshow(total_return[index,:,:],cmap='coolwarm')
			plt.colorbar()

			plt.subplot(233)
			plt.title('Average Profit')
			plt.xlabel('Threshold')
			plt.ylabel('K')
			plt.xticks(np.arange(0, 51, step=10))
			plt.yticks(np.arange(0, 11, step=5), ('10', '50', '100'))
			plt.imshow(total_return[index,:,:]/trade_time[index,:,:],cmap='coolwarm')
			plt.colorbar()

			plt.subplot(234)
			plt.title('Profit with Cost bp1')
			plt.xlabel('Threshold')
			plt.ylabel('K')
			plt.xticks(np.arange(0, 51, step=10))
			plt.yticks(np.arange(0, 11, step=5), ('10', '50', '100'))
			plt.imshow(total_return[index,:,:]-trade_time[index,:,:]*1,cmap='coolwarm')
			plt.colorbar()

			plt.subplot(235)
			plt.title('Profit with Cost bp2')
			plt.xlabel('Threshold')
			plt.ylabel('K')
			plt.xticks(np.arange(0, 51, step=10))
			plt.yticks(np.arange(0, 11, step=5), ('10', '50', '100'))
			plt.imshow(total_return[index,:,:]-trade_time[index,:,:]*2,cmap='coolwarm')
			plt.colorbar()

			plt.subplot(236)
			plt.title('Profit with Cost bp3')
			plt.xlabel('Threshold')
			plt.ylabel('K')
			plt.xticks(np.arange(0, 51, step=10))
			plt.yticks(np.arange(0, 11, step=5), ('10', '50', '100'))
			plt.imshow(total_return[index,:,:]-trade_time[index,:,:]*3,cmap='coolwarm')
			plt.colorbar()

			#plt.show()
			plt.savefig('day{}.png'.format(index+1))
			plt.clf()



if __name__ == '__main__':
	#plot_by_k()
	#plot_by_threshold()
	plot_by_day()
