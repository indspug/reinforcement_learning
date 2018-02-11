# coding: utf-8

import numpy as np

##################################################
# シグモイド関数
##################################################
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

##################################################
# 二乗誤差
##################################################
def mean_squared_error(y, t):

	n = y.shape[0]
	loss = 0.5 * np.sum((y - t)**2)
	return loss / n

##################################################
# クロスエントロピー
##################################################
def cross_entropy(y, t):
	
	delta = 1e-7
	
	# yが1次元の場合は、2次元データに変換
	if y.ndim == 1:
		y = y.reshape(1, y.size)
		t = t.reshape(1, t.size)
	
	# 教師データがラベルの場合、
	# one-hot-vectorに変換
	if t.size != y.size:
		t1 = np.zeros_like(y)
		for i in range(t.shape[0]):
			maxidx = t[i]
			t1[i,maxidx] = 1
		t = t1
	
	n = y.shape[0]
	loss = -np.sum(t * np.log(y+delta))
	
	return (loss / n)

##################################################
# ソフトマックス関数
##################################################
def softmax(x):
	
	# MNISTの場合：x.shape=(N, 10)
	if x.ndim == 2:
		# オーバーフロー防止のために最大値を引く
		x_max = np.max(x, axis=1)
		x_max = x_max[:, np.newaxis]
		x = x - x_max
		
		x_sum = np.sum(np.exp(x), axis=1)	# (N,)
		x_sum = x_sum[:, np.newaxis]		# (N,1)
		y = np.exp(x) / x_sum
		return y
		
		# テキストに書いてあったのは下記
		#x = x.T
		#x = x - np.max(x, axis=0)
		#y = np.exp(x) / np.sum(np.exp(x), axis=0)
		#return y.T
		
		
	# そのまま計算するとオーバーフローするので
	# 最大値を引いた値でexpの計算を行う
	if x.ndim == 1:
		c = np.max(x)
		exp_a = np.exp(x - c)
		sum_exp_a = np.sum(exp_a)
		y = exp_a / sum_exp_a
	
	return y

##################################################
# ReLU
##################################################
def relu(x):
	return np.maximum(0, x)

