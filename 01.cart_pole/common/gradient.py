# coding: utf-8

import numpy as np

##################################################
# 勾配
##################################################
def numerical_gradient(f, x):
	
	h = 1e-4	# 0.0001
	grad = np.zeros_like(x)
	
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		idx = it.multi_index
		save_val = x[idx]
		
		x[idx] = float(save_val) - h
		fxh1 = f(x)	# f(x-h)
		
		x[idx] = float(save_val) + h
		fxh2 = f(x)	# f(x+h)
		
		grad[idx] = (fxh2 - fxh1) / (2*h)	# 勾配
		x[idx] = save_val					# xの値を元に戻す
		it.iternext()						# イテレータを次に進める
	
	return grad

