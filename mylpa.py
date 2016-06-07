#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import random
import os

def fitlabel(label_count):
	fit_label = []
	label_num = 0
	for label in label_count.index:
		if label_count[label] > label_num:
			label_num = label_count[label]
			fit_label = [label]
		elif label_count[label] == label_num:
			fit_label.append(label)
	return fit_label

def lpa(connection, thre):
	users = connection.keys()
	labels = pd.Series(index = users, data = users)
	total_num = len(users)
	loop_num = 0
	while True:
		change_num = 0
		for i in range(total_num):
			user = users[i]
			relationships = connection[user]
			label_count = pd.Series()
			for co_user in relationships:
				label = labels[co_user]
				if label not in label_count.index.values:
					label_count.loc[label] = 1
				else:
					label_count[label] += 1
			temp_label = random.choice(fitlabel(label_count))
			if labels[user] != temp_label:
				change_num += 1
			labels[user] = temp_label
		loop_num += 1
		print loop_num
		if float(change_num)/float(total_num) <= 1-thre:
			print 'finally, we looped %d times......'%loop_num
			break
	return labels

def test():
	connection = {1:[2,5,6], 2:[1,3,6], 3:[2,4,12], 4:[3,5,6,7], 5:[1,4,6], 6:[1,2,4,5], 
					7:[4,8,11,12], 8:[7,9,12], 9:[8,10,12], 10:[9,11,12], 11:[7,10,12], 12:[7,8,9,10,11]}
	thre = 1
	print lpa(connection, thre)

def main():
	column = ['userid', 'movieid', 'rating', 'timestamp']
	ratings = pd.read_table('ml-1m/ratings.dat', sep = '::', names = column)
	user_movie = ratings.pivot(index = 'userid', columns = 'movieid', values = 'rating')

	lucky_user = np.random.permutation(user_movie.index.values)[:200]
	user_num = len(user_movie.index)
	t = user_num / 10
	hot_movie = user_movie.count()[user_movie.count()<t].index

	using_data = user_movie.loc[lucky_user, hot_movie].dropna(how = 'all').dropna(axis = 1, how = 'all')
	print using_data.shape
	connection = {}
	for user in using_data.index:
		connection[user] = np.array([], dtype = 'int')
	for movie in using_data.columns:
		love_movie = using_data[movie][using_data[movie]>=3].index.values
		hate_movie = using_data[movie][using_data[movie]<3].index.values
		for user in love_movie:
			other_love_user = list(set(love_movie)-set([user]))
			connection[user] = np.append(connection[user], other_love_user)
		for user in hate_movie:
			other_hate_user = list(set(hate_movie)-set([user]))
			connection[user] = np.append(connection[user], other_hate_user)
	print len(connection.keys())
	thre = 1
	labels = lpa(connection, thre)
	print labels
	print set(labels.values)
	print len(set(labels.values))

if __name__ == '__main__':
	test()