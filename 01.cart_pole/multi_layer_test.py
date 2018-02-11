# -*- coding: utf-8 -*-

import numpy as np
import sys, os
sys.path.append(os.pardir)
from multi_layer_net import MultiLayerNet
from common.optimizer import *

##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	train_size = 100000
	batch_size = 100
	
	x = np.zeros((train_size, 4))
	t = np.zeros((train_size, 2))
	for i in range(train_size):
		x1 = np.random.rand()*10
		x2 = np.random.rand()*10
		x3 = np.random.rand()*10
		x4 = np.random.rand()*10
		t1 = (x1 + x2 + x3 + x4)**2
		t2 = (x1 + x2 + x3 + x4)**3
		x[i,0] = x1
		x[i,1] = x2
		x[i,2] = x3
		x[i,3] = x4
		t[i,0] = t1
		t[i,1] = t2
		#if i==100:
		#	print('(%.2f,%.2f,%.2f,%.2f)->(%.2f,%.2f)' % (x1,x2,x3,x4,t1,t2))
	
	network = MultiLayerNet(input_size=4, hidden_size_list=[8,16], \
							output_size=2)
	optimizer = Adam(lr=0.001)
	
	for i in range(10000000):
		# ミニバッチの取得
		batch_idx = np.random.choice(train_size, batch_size)
		x_batch = x[batch_idx]
		t_batch = t[batch_idx]
		
		grads = network.gradient(x_batch, t_batch)
		optimizer.update(network.params, grads)
		
		if (i % 1000) == 0:
			loss = network.loss(x_batch, t_batch)
			print('step %06d: loss=%.4f' % (i/1000, loss) )
			x_test = np.array([[5,6,7,8],[1,2,3,4]])
			y_test = network.predict(x_test)
			print('  y = %s' % y_test)
	
	
