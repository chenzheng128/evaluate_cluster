#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import os

'''
Use this pyfile to create predictedratings.dat file
'''

def CollaborativeFiltering(init_rating_data):
	predicted_data = init_rating_data.copy()

	print "Computing the correlation, please wait......"
	corr = init_rating_data.T.corr(min_periods=100)
	corr_clean = corr.dropna(how='all')					#drop nan data by index
	corr_clean = corr_clean.dropna(axis=1, how='all')	#drop nan data by column
	predicted_data = predicted_data.loc[corr_clean.index]

	print "Predicting the unrated movies, please wait......"
	total_user = len(corr_clean.index)
	proceeding = 0
	for user in corr_clean.index.values:
		proceeding += 1
		if proceeding%10==0:
			print 'Computing NO.', proceeding, 'user/total number is', total_user
		noRating = init_rating_data.ix[user]
		noRating = noRating[noRating.isnull()]			#movies that user never rated before
		corr_user = corr_clean[user].drop(user)			#drop user itself
		corr_user = corr_user[corr_user>0.2].dropna()	#pick corr_user that the correlation>0.2
		for movie in noRating.index:
			prediction = []
			for other in corr_user.index:
				if not np.isnan(init_rating_data.ix[other, movie]):
					prediction.append((init_rating_data.ix[other, movie], corr_clean[user][other]))
			if prediction:
				predicted_data[movie][user] = sum([value*weight for value, weight in prediction])/sum([pair[1] for pair in prediction])

	print "Writing predicted data into dat file, please wait......"
	res_path = os.path.join(os.getcwd(), '../ml-1m/predictedratings_2.dat')
	predicted_data.to_csv(res_path)

	print "Done, please check your predictedratings_2.dat in"+os.path.join(os.getcwd(), 'ml-1m')
	print predicted_data.shape
	return predicted_data
	
def main():
	localPath = os.getcwd()
	srcPath = os.path.join(localPath, '../ml-1m/smallratings.dat')
	print 'srcPath = ', srcPath
	init_rating_data = pd.read_table(srcPath, sep = ',', header = 0, index_col = 0)

	predicted_rating_data = CollaborativeFiltering(init_rating_data)

if __name__ == '__main__':
		main()	