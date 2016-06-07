#!/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import sys, os

def main():
	data_path = 'ml-1m/ratings.dat'
	columns = ['userid', 'movieid', 'rating', 'timestamp']
	ratings = pd.read_table(data_path, sep = '::', header = None, names = columns, engine='python')
	rating_data = ratings.pivot(index = 'userid', columns = 'movieid', values = 'rating')
	#print rating_data
	rating_num = rating_data.T.count()
	print rating_num.count()
	movie_num = [0]*31
	print 'Computing......'
	for num in rating_num:
		place = num/10
		if place >= 30:
			movie_num[30] += 1
		else:
			movie_num[place] += 1
	print movie_num
	return 0

if __name__ == '__main__':
	main()