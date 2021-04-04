import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import psycopg2, psycopg2.extras
from statistics import mean


def weird_division(n, d):
	return n / d if d else 0

def merge(folder):
	tensor = []
	files = os.listdir(folder)
	for i in range(len(files)):
		if os.path.isfile('{}/{}.npy'.format(folder, i)):
			mat = np.load('{}/{}.npy'.format(folder, i))
		tensor.append(mat)
	tensor = np.array(tensor)

	np.save('{}.npy'.format(folder), tensor)


def main():
	merge('total_return')
	merge('trade_time')

if __name__ == '__main__':
	main()