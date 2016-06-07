#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import os

'''
Use this file to cut data and create small data to analysis
'''

def main():
	src_path = '../ml-1m/ratings.dat'
	columns = ['userid', 'movieid', 'rating', 'timestamp']
	ratings = pd.read_table(src_path, sep = '::', header = None, names = columns)
	user_movie_data = ratings.pivot(index = 'userid', columns = 'movieid', values = 'rating')
	select_movies = user_movie_data.count().order(ascending = False).index.values[100:2100]
	user_cutmovie_data = user_movie_data.loc[ : , select_movies]
	select_users = user_cutmovie_data.count(axis = 1).order(ascending = False).index.values[ :1200]
	cut_data = user_cutmovie_data.loc[select_users]
	print cut_data.shape
	print cut_data.count().order(ascending = False)
	print cut_data.count(axis = 1).order(ascending = False)
	cut_data.to_csv('../ml-1m/smallratings.dat')


if __name__ == '__main__':
	main()