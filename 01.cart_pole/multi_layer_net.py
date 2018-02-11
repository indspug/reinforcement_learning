# coding: utf-8

import sys, os
sys.path.append(os.getcwd())
#from common.functions import *
from common.gradient import numerical_gradient
from common.layers import Affine, Sigmoid, ReLU, SoftmaxWithLoss, IdentifyWithLoss

import numpy as np
from collections import OrderedDict

##################################################
# 複数層のニューラルネットワーク
##################################################
class MultiLayerNet:
	
	"""
		input_size: 入力サイズ(MNISTの場合は784)
		hidden_size_list: 隠れ層のニューロンの数(e.g. [100,100,100])
		output_size: 出力サイズ(MNISTの場合は10)
		activation: 'relu' or 'sigmoid'
	"""
	# ------------------------------
	# コンストラクタ
	# ------------------------------
	def __init__(self, input_size, hidden_size_list, output_size, 
					activation='relu', weight_init_std='relu', weight_decay_lambda=0):
		
		self.input_size = input_size
		self.hidden_size_list = hidden_size_list
		self.hidden_layer_num = len(hidden_size_list)
		self.output_size = output_size
		self.weight_decay_lambda = weight_decay_lambda
		self.params = {}
		
		# 重みの初期化
		self.__init_weight(weight_init_std)
		
		# レイヤの生成
		activation_layer = {'sigmoid': Sigmoid, 'relu': ReLU}
		self.layers = OrderedDict(	)	# 順番付ディクショナリ
		for i in range(1, self.hidden_layer_num+1):
			
			# 隠れ層1〜hidden_layer_numの生成
			no = str(i)
			self.layers['Affine'+no] = Affine(self.params['W'+no], self.params['b'+no])
			
			# 活性化関数
			self.layers['Func'+no] = activation_layer[activation]()
			
		# 出力層
		no = str(self.hidden_layer_num+1)
		self.layers['Affine'+no] = Affine(self.params['W'+no], self.params['b'+no])
		#self.lastLayer = SoftmaxWithLoss()
		self.lastLayer = IdentifyWithLoss()
	
	# ------------------------------
	# 重みの初期化
	# ------------------------------
	def __init_weight(self, weight_init_std):
		"""
			weight_init_std: 重みの標準偏差(e.g. 0.01)
				'relu'または'he'を指定した場合はHeの初期値を設定、
				'sigmoid'または'xavier'を指定した場合はXavierの初期値を設定
		"""
		layer_size_list = [self.input_size] + self.hidden_size_list + [self.output_size];
		for i in range(1, len(layer_size_list)):
			# 標準偏差の設定
			scale = weight_init_std
			scale_std = str(weight_init_std).lower()
			if scale_std in ('relu', 'he'):
				scale = np.sqrt(2.0 / layer_size_list[i-1])
			elif scale_std in ('sigmoid', 'xavier'):
				scale = np.sqrt(1.0 / layer_size_list[i-1])
			
			# 重みを標準偏差=scaleの正規分布で初期化
			no = str(i)
			size1 = layer_size_list[i-1]
			size2 = layer_size_list[i]
			self.params['W'+no] = scale * np.random.randn(size1, size2)
			self.params['b'+no] = np.zeros(size2)
		
	# ------------------------------
	# 認識
	# ------------------------------
	def predict(self, x):
		y = x
		for key, layer in self.layers.items():
			y = layer.forward(y)
		
		return y
	
	# ------------------------------
	# 損失関数
	# ------------------------------
	def loss(self, x, t):
		
		y = self.predict(x)
		
		# Weight Decay : 損失に(0.5 * λ * W^2)を足す
		weight_decay = 0
		for i in range(0, self.hidden_layer_num+1):
					# 出力層も含む
			W = self.params['W' + str(i+1)]
			weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
			#print(weight_decay)
		
		# 二乗誤差
		#loss = 0.5 * np.sum((y - t)**2, axis=1) + weight_decay
		#loss = np.sum(loss) / float(t.shape[0])
		loss = self.lastLayer.forward(y, t) + weight_decay;
		return loss
	
	# ------------------------------
	# 認識精度
	# ------------------------------
	def accuracy(self, x, t):
		y = self.predict(x)
		y = np.argmax(y, axis=1)
		t = np.argmax(t, axis=1)
		n = x.shape[0]
		accuracy = np.sum(y==t) / float(n)
		return accuracy
		
	# ------------------------------
	# 勾配(誤差逆伝播法)
	# ------------------------------
	def gradient(self, x, t):
		
		# 順伝播計算
		loss = self.loss(x, t)
		
		# 逆伝播計算
		dout = 1
		#batch_size = x.shape[0]
		#dout = (self.y - t) / batch_size
		dout = self.lastLayer.backward(dout)
		
		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)
		
		# 勾配設定
		grads = {}
		for i in range(0, self.hidden_layer_num+1):
					# 出力層も含む
			# Weight Decayを使用する場合はWにλ*Wを足す
			no = str(i+1)
			grads['W'+no] = self.layers['Affine'+no].dW + \
							self.weight_decay_lambda * self.params['W'+no];
			grads['b'+no] = self.layers['Affine'+no].db
			
		return grads
	
