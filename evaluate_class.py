#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import os, logging
from sklearn.cluster import KMeans
from sklearn.cluster import Ward
from sklearn.cluster.bicluster import SpectralBiclustering
from sklearn.cluster import spectral_clustering

import pyfile.evaluate_cf as ecf

class evaluate:
	def __init__(self, ratings, data_matrix, method, k):
		#类的初始化，需要参数：ratings, data_matrix, method, k
		#	ratings: 最原始数据集
		#	data_matrix: 使用数据的数据矩阵
		#	method: 使用的聚类方式
		#	k: 聚类个数
		self.ratings = ratings
		self.data_matrix = data_matrix
		self.k = k
		if method=='km':
			result = self.__kmclu()
		elif method=='hc':
			result = self.__hieclu()
		elif method=='sc':
			result = self.__speclu()
		else:
			print 'There is no method called %s'%(method)
			result = []
		self.result = result

	def __kmclu(self):
		#use k-means
		print 'using k-means clustering......'
		km = KMeans(n_clusters = self.k)
		km.fit(self.data_matrix)
		result = km.predict(self.data_matrix)
		return result

	def __hieclu(self):
		#use Hierarchical clustering
		print 'using hierarchical clustering......'
		ac = Ward(n_clusters = self.k)
		ac.fit(self.data_matrix)
		result = ac.fit_predict(self.data_matrix)
		return result

	def __getEMatrix(self):
		'''构建二部图邻接矩阵E
		'E = [[0 A],
		'	[transpose(A) 0]],
		'其中A为二部图的N×M矩阵，N为总user数量，M为总电影数量
		'''
		data_matrix = self.data_matrix
		userNum, movieNum = data_matrix.shape
		t_l_matrix = np.zeros((userNum, userNum))
		t_r_matrix = data_matrix
		b_l_matrix = data_matrix.transpose()
		b_r_matrix = np.zeros((movieNum, movieNum))
		top_matrix = np.hstack((t_l_matrix, t_r_matrix))
		bottom_matrix = np.hstack((b_l_matrix, b_r_matrix))
		E_matrix = np.vstack((top_matrix, bottom_matrix))
		return E_matrix

	def __speclu(self):
		#use spectral clustering
		print 'using spectral clustering......'
		data_matrix = self.data_matrix
		if len(data_matrix) == len(data_matrix[0]):
			print "Donot need to use E_matrix"
			E_matrix = data_matrix
		else:
			E_matrix = self.__getEMatrix()
		result_total = spectral_clustering(E_matrix, n_clusters = self.k)
		result = result_total[ : len(data_matrix)]
		return result

	def __evaluateCF(self, rating_data, thre):
		#类的内置函数，用来计算社区中的协同评价
		#	rating_data: 社区数据
		#	thre: 相关用户系数阈值
		check_data = rating_data.copy()
		# Extrack real ratings to evaluate collaborative filtering ratings
		extrack_ratings = pd.DataFrame(columns = ['movieid', 'rating'])
		for user in check_data.index:
			movieid = np.random.choice(check_data.loc[user].dropna().index.values)
			rating = check_data.loc[user, movieid]
			extrack_ratings.loc[user] = [movieid, rating]
			check_data.loc[user, movieid] = np.nan
		#print "Computing the correlation, please wait......"
		corr = check_data.T.corr(min_periods = 50)
		predicted_ratings = pd.DataFrame(index = extrack_ratings.index, columns = extrack_ratings.columns)
		#print "Predicting the unrated movies, please wait......"
		for user in extrack_ratings.index:
			movie = extrack_ratings.movieid[user]
			corr_user = corr[user].drop(user)				#drop user itself
			corr_user = corr_user[corr_user > thre].dropna()	#pick corr_user that the correlation>0.1
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

	def EvaCF(self, thre, n):
		#协同预测评价社区质量
		#	thre: 相关用户系数阈值
		#	n: 评价次数
		print 'evaluating......'
		ratings = self.ratings
		result = self.result
		evaluate_results = pd.DataFrame()
		evaluate_time = n
		for i in range(evaluate_time):
			evaluate_result = pd.Series()
			userid = ratings.index.values
			part = 0
			for cls in set(result):
				part += 1
				cls_index = [userid[x] for x in range(len(userid)) if result[x]==cls]
				cls_rating_data = ratings.loc[cls_index]
				#print ('The %d part in the %dth evaluate')%(part, i+1)
				cls_evaluate_result = self.__evaluateCF(cls_rating_data, thre)
				evaluate_result = evaluate_result.append(cls_evaluate_result)
			evaluate_results[i] = evaluate_result
		#print evaluate_results.describe()
		print 'Total : ', evaluate_results.mean().describe()
		#logging.debug(evaluate_results.mean().describe())

	def __movie_genres(movie_genres_data):
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

	def EvaClu(self, user_genres_data, movie_genres_data, top_n = 5):
		#评价社区质量在电影种类方面的特征(论文中未用到)
		#	user_genres_data: 用户-电影种类矩阵
		#	movie_genres_data: 电影-种类矩阵
		#	top_n: 选取用户最喜爱前5种电影作电影种类分析
		print 'evaluate cluster result......'
		ratings = self.ratings
		#由于电影本身种类是不均衡的，故引入movie_genres_col参数
		#该参数是每个电影种类占总种类的百分比
		movie_genre_dict, genres = self.__movie_genres(movie_genres_data)
		genres_num = pd.Series(index = list(genres), data = [0]*len(genres))
		for movie in movie_genre_dict.keys():
			for genre in movie_genre_dict[movie]:
				genres_num[genre] += 1
		movie_genres_col = (genres_num/genres_num.sum()) ** 0.8
		#print movie_genres_col
		userid = ratings.index.values
		genres = user_genres_data.columns
		class_genres_data = pd.DataFrame(index = set(result), columns = genres)
		class_genres_data = class_genres_data.fillna(0)
		n = top_n
		for user_index in range(len(result)):
			user_id = userid[user_index]
			user_class = result[user_index]
			#根据movie_genres_col参数 均衡电影种类个数
			balance_user_data = user_genres_data.loc[user_id]/movie_genres_col
			faver_sort = balance_user_data.order(ascending = False)
			score = 1
			for genre in faver_sort.index.values[ : n]:
				#score -= 1
				class_genres_data.loc[user_class, genre] += score
		print class_genres_data
		class_genres_co = class_genres_data.copy()
		for class_id in class_genres_data.index:
			score = class_genres_data.loc[class_id].sum()
			class_genres_co.loc[class_id] = (class_genres_data.loc[class_id]/score) * 100
		print class_genres_co
		print class_genres_co.describe().loc['std'].sum()
		class_genres_co.to_csv('clustering.csv')
		top_genres = {}
		for clu in class_genres_co.index:
			top_genres[clu] = class_genres_co.loc[clu].order(ascending = False)[ : 5].index.values
		print top_genres
		print 'Done.'

	def __ConnectNumber(self, cls_ratings):
		connect = 0.0
		for movie in cls_ratings.columns:
			n = cls_ratings[movie].count()
			connect += (n*n - n)/2
		return connect

	def __getO(self, cls_rating_data):
		ratings = self.ratings
		connect = 0.0
		InCommunityUsers = set(cls_rating_data.index.values)
		OutCommunityUsers = set(ratings.index.values) - InCommunityUsers
		OutRatings = ratings.loc[OutCommunityUsers]
		for movie in ratings.columns:
			iu = cls_rating_data[movie].count()
			ou = OutRatings[movie].count()
			connect += iu*ou
		return connect

	def Q(self):
		#利用传统模块度Q值评价社区质量(计算出的Q值特别小，论文中未用到)
		ratings = self.ratings
		result =self.result
		E = self.__ConnectNumber(ratings)
		userid = ratings.index.values
		Q = 0.0
		for cls in set(result):
			cls_index = [userid[x] for x in range(len(userid)) if result[x]==cls]
			print 'community size : ', len(cls_index)
			cls_rating_data = ratings.loc[cls_index]
			I = self.__ConnectNumber(cls_rating_data)
			O = self.__getO(cls_rating_data)
			Q += (I/E - ((2*I+O)/(2*E))**2)
		print Q

def LoadData():
	#加载数据
	print 'loading data......'
	local_path = os.getcwd()

	ug_src_path = os.path.join(local_path, 'ml-1m/user_genres_data.csv')
	user_genres_data = pd.read_csv(ug_src_path, header = 0, index_col = 0)

	movie_src_path = 'ml-1m/movies.dat'
	columns_1 = ['movieid', 'titles', 'genres']
	movie_genres_data = pd.read_table(movie_src_path, sep = '::', header = None, index_col = 0, encoding = 'gbk', names = columns_1)

	predictedrating_src_path = os.path.join(local_path, 'ml-1m/predictedratings.dat')
	predictedratings = pd.read_table(predictedrating_src_path, sep = ',', header = 0, index_col = 0)
	predictedratings.columns = [int(x) for x in predictedratings.columns]

	rating_src_path = os.path.join(local_path, 'ml-1m/smallratings.dat')
	ratings = pd.read_table(rating_src_path, sep = ',', header = 0, index_col = 0)
	int_col = [int(x) for x in ratings.columns]
	ratings.columns = int_col
	return ratings, predictedratings, user_genres_data, movie_genres_data

def filldata(ratings):
	'''
	#实现均值填充功能
	'''
	fill_ratings = ratings.copy()
	ratings_mean = ratings.mean()
	for movie in ratings.columns:
		fill_ratings[movie] = ratings[movie].fillna(ratings_mean[movie])
	return fill_ratings

def main():
	#Step 1 : Load data
	ratings, predictedratings, user_genres_data, movie_genres_data = LoadData()
	#由于数据中有很多NAN值，所以需要先做填充(零值填充、均值填充、协同预测填充)
	#进行数据评价的时候选择需要用到的数据即可
	rating_data = ratings.fillna(0)							#零值填充数据
	fill_rating_data = filldata(ratings)					#均值填充数据
	predictedrating_data = predictedratings.fillna(0)		#协同预测填充数据
	fill_predictedrating_data = filldata(predictedratings)	#协同预测+均值填充数据

	#用户-电影种类矩阵
	genres = user_genres_data.columns.values
	user_genres_col = user_genres_data.copy()
	for user in user_genres_data.index:
		total = user_genres_data.loc[user].sum()
		user_genres_col.loc[user] = user_genres_data.loc[user]/total

	'''
	#用户-电影种类矩阵中删除热门电影，论文中没有用到
	hot_topic = user_genres_col.sum().order(ascending = False)[:2].index.values
	user_genres_col = user_genres_col.drop(hot_topic, axis = 1)

	#用户-用户矩阵
	print 'Computing user_corr data, please wait ......'
	user_corr = ratings.T.corr(min_periods = 1)
	user_corr = user_corr + 1
	user_corr = user_corr.fillna(1)
	#user_corr = user_corr[user_corr>0].fillna(0)
	print user_corr
	'''

	#Step 2 : 选取一个数据集，做聚类
	#删除热门电影
	user_num = len(ratings.index)
	t = user_num * 0.3
	target_movie = ratings.count()[ratings.count()<t].index
	print len(target_movie)
	#using_data 是选取的数据集
	using_data = fill_predictedrating_data[target_movie].copy()
	#using_data = user_corr.copy()

	#Step 3 : Choice one cluster algorithm to do cluster
	print 'clustering......'
	#cluster
	data_matrix = using_data.values
	methods = ['km', 'hc', 'sc']
	for method in methods:
		#method = methods[1]
		k = 6
		evl = evaluate(ratings, data_matrix, method, k)
		evl.Q()
		evl.EvaCF(0.1, 10)

	return 0

if __name__ == '__main__':
	main()