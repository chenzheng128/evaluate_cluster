#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import os

def evaluateCF(rating_data, thre):
	check_data = rating_data.copy()
	#check_data = check_data.loc[check_data.count(axis = 1)>200]
	# Extrack real ratings to evaluate collaborative filtering ratings
	extrack_ratings = pd.DataFrame(columns = ['movieid', 'rating'])
	for user in check_data.index:
		movieid = np.random.choice(check_data.loc[user].dropna().index.values)
		rating = check_data.loc[user, movieid]
		extrack_ratings.loc[user] = [movieid, rating]
		check_data.loc[user, movieid] = np.nan

	#print "Computing the correlation, please wait......"
	corr = check_data.T.corr(min_periods = thre)
	predicted_ratings = pd.DataFrame(index = extrack_ratings.index, columns = extrack_ratings.columns)
	#print "Predicting the unrated movies, please wait......"
	for user in extrack_ratings.index:
		movie = extrack_ratings.movieid[user]
		corr_user = corr[user].drop(user)				#drop user itself
		corr_user = corr_user[corr_user>0.1].dropna()	#pick corr_user that the correlation>0.1
		prediction = []
		for other in corr_user.index:
			if not np.isnan(check_data.loc[other, movie]):
				prediction.append((check_data.loc[other, movie], corr[user][other]))
		if prediction:
			rating = sum([value*weight for value, weight in prediction])/sum([pair[1] for pair in prediction])
			predicted_ratings.loc[user] = [movie, rating]
	residue_user = predicted_ratings['rating'].dropna().index.values
	evaluate_result = (extrack_ratings['rating'][residue_user] - predicted_ratings['rating'][residue_user]).abs()
	return evaluate_result.astype(float)