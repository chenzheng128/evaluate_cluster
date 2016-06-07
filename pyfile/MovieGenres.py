#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import os

def movie_genres(movie_genres_data):
	'''Create a dict which keys are movieids and values are movie genres

		Return movie_genre_dict(type is dict) and genre(type is set)
	'''
	movie_genre_dict = {}
	genres = set()
	for movieid, title, genre in movie_genres_data.itertuples():
		this_genre = genre.split('|')
		movie_genre_dict[movieid] = this_genre
		genres = genres | set(this_genre)
	return movie_genre_dict, genres

def user_genres(rating_data, movie_genre_dict, genres):
	'''Create a DataFrame which:
		index are userids,
		columns are genres,
		values are numbers of genres that user has rated

		Return user_genres_data(type is DataFrame)
	'''
	user_genres_data = pd.DataFrame(index = rating_data.index, columns = genres)
	user_genres_data = user_genres_data.fillna(0)
	for userid in rating_data.index.values:
		for movieid in rating_data.loc[userid].dropna().index:
			for genre in movie_genre_dict[movieid]:
				user_genres_data.loc[userid, genre] += 1
	print user_genres_data
	user_genres_data.to_csv('ml-1m/user_genres_data.csv')
	return user_genres_data

def main():
	local_path = os.getcwd()
	movie_src_path = os.path.join(local_path, 'ml-1m/movies.dat')
	columns_1 = ['movieid', 'titles', 'genres']
	movie_genres_data = pd.read_table(movie_src_path, sep = '::', header = None, index_col = 0, encoding = 'gbk', names = columns_1)
	#print movie_genres_data
	rating_src_path = os.path.join(local_path, 'ml-1m/smallratings.dat')
	rating_data = pd.read_table(rating_src_path, sep = ',', header = 0, index_col = 0)
	int_col = [int(x) for x in rating_data.columns]
	rating_data.columns = int_col

	movie_genre_dict, genres = movie_genres(movie_genres_data)
	#print movie_genre_dict, len(genres)
	user_genres_data = user_genres(rating_data, movie_genre_dict, genres)

if __name__ == '__main__':
	main()