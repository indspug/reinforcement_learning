# coding: utf-8

import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.util import *
import numpy as np

##################################################
# ReLU
##################################################
class ReLU:
	
	# コンストラクタ
	def __init__(self):
		self.mask = None
	
	# 順伝播
	def forward(self, x):
		self.mask = (x <= 0)	# 0より大きいならFalse
		fx = x.copy()
		fx[self.mask] = 0		# True(0以下)の要素は不活性(0)にする
		return fx
	
	# 逆伝播
	def backward(self, dout):
		# f'(x) = 1(x>0), 0(x<=0)
		dfx = dout.copy()
		dfx[self.mask] = 0
		return dfx
	
##################################################
# シグモイド関数
##################################################
class Sigmoid:
	
	# コンストラクタ
	def __init__(self):
		self.fx = None
	
	# 順伝播
	def forward(self, x):
		fx = sigmoid(x)
		self.fx = fx
		return fx
	
	# 逆伝播
	def backward(self, dout):
		# f'(x) = (1 - f(x)) * f(x)
		dfx = dout * (1.0 - self.fx) * self.fx
		return dfx
	
##################################################
# アフィン変換
##################################################
class Affine:
	
	# コンストラクタ
	def __init__(self, W, b):
		self.W = W
		self.b = b
		
		self.x = None
		self.original_x_shape = None
		
		self.dW = None
		self.db = None
	
	# 順伝播
	def forward(self, x):
		# 
		self.original_x_shape = x.shape
		x = x.reshape(x.shape[0], -1)
		self.x = x
		
		# u = x * W + b		(N,1,j) * (j,i) -> (N,i)
		#					ex. (100,1,784) * (784,50) -> (100, 50)
		u = np.dot(self.x, self.W) + self.b
		return u
		
	# 逆伝播
	def backward(self, delta):
		# dL/du = dL/du(l+1) * W.T
		dx = np.dot(delta, self.W.T)
		# dL/dW = dL/du * W.T
		self.dW = np.dot(self.x.T, delta)
		# dL/db = dL/du
		self.db = np.sum(delta, axis=0)
		
		#delta = np.dot(delta, self.W.T)
		dx = dx.reshape(*self.original_x_shape)
		return dx

##################################################
# 恒等写像(出力層)
##################################################
class IdentifyWithLoss:
	
	# コンストラクタ
	def __init__(self):
		self.loss = None
		self.y = None	# softmaxの出力
		self.t = None	# 教師データ
	
	# 順伝播
	def forward(self, x, t):
		self.t = t
		self.y = x
		self.loss = mean_squared_error(self.y, self.t)
		return self.loss
		
	# 逆伝播
	def backward(self, dout=1):
		N = self.t.shape[0]
		delta = (self.y - self.t) / N
		return delta

##################################################
# ソフトマックス(出力層)
##################################################
class SoftmaxWithLoss:
	
	# コンストラクタ
	def __init__(self):
		self.loss = None
		self.y = None	# softmaxの出力
		self.t = None	# 教師データ
	
	# 順伝播
	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy(self.y, self.t)
		return self.loss
		
	# 逆伝播
	def backward(self, dout=1):
		N = self.t.shape[0]
		if self.t.size == self.y.size:
			delta = (self.y - self.t) / N
		else:
			delta = self.y.dopy()
			delta[np.arange(N), self.t] -= 1
			delta = delta / N
		
		return delta

##################################################
# BatchNormalization
##################################################
class BatchNormalization:
	
	# コンストラクタ
	def __init__(self, gamma, beta, momentum=0.9, \
					running_mean=None, running_var=None):
		self.gamma = gamma
		self.beta = beta
		self.momentum = momentum
		self.input_shape = None
		
		# テスト時に使用する平均と分散
		self.running_mean = running_mean
		self.running_var  = running_var
		
		# 逆伝播(backward)時に使用する中間データ
		self.batch_size = None
		self.xc = None
		self.std = None
		self.dgamma = None
		self.dbeta = None
		
	# 順伝播
	def forward(self, x, train_flg=True):
		self.input_shape = x.shape
		
		# xを(N,)形式に変換(畳み込み計算用)
		if x.ndim != 2:
			N, C, H, W = x.shape
			x = x.reshape(N, -1)	# (N, C*H*W)
		
		out = self.__forward(x, train_flg)
		
		# 形式を入力と同じに戻してReturn
		return out.reshape(*self.input_shape)
		
	# 順伝播計算
	def __forward(self, x, train_flg):
		
		if self.running_mean is None:
			N, D = x.shape
			self.running_mean = np.zeros(D)
			self.running_var  = np.zeros(D)
		
		if train_flg:
			# 学習時
			mu = x.mean(axis=0)		# μ = 1/N * Σx
			xc = x - mu				# xc = x - μ
			var = np.mean(xc**2, axis=0)	# σ^2 = Σ(x-μ)^2
			std = np.sqrt(var + 10e-7)		# σ^2 - ε
			xn = xc / std	# x^ = (x - μ) / √(σ^2 - ε)
			
			self.batch_size = x.shape[0]
			self.xc = xc
			self.xn = xn
			self.std = std
			self.running_mean = self.momentum * self.running_mean + \
								(1 - self.momentum) * mu
			self.running_var  = self.momentum * self.running_var + \
								(1 - self.momentum) * var
		else:
			# テスト時
			xc = x - self.running_mean
			xn = xc / (np.sqrt(self.running_var + 10e-7))
		
		out = self.gamma * xn + self.beta
		return out
	
	# 逆伝播
	def backward(self, dout=1):
		
		# δを(N,)形式に変換(畳み込み計算用)
		if dout.ndim != 2:
			N,C,H,W = dout.shape
			dout = dout.reshape(N,-1)	# (N,C*H*W)
		
		# 元の形式に戻してReturn
		delta = self.__backward(dout)
		delta = delta.reshape(*self.input_shape)
		
		return delta
		
	# 逆伝播計算
	def __backward(self, dout):
		# y = (γ * x^) + β
		dbeta = dout.sum(axis=0)	# dβ = Σdout
		dgamma = np.sum(self.xn * dout, axis=0)	# dγ = Σ(dout * x^)
		dxn = self.gamma * dout		# dx^ = γ * dout
		
		# x^ = (x - μ) / √(σ^2 - ε)
		dxc = dxn / self.std
			# d(x-μ) = dx^ / √(σ^2 - ε)
		
		# x^ = (x - μ) * t 
		#		(t = 1 / √(σ^2 - ε))
		# t = 1 / u
		#		(u = √(σ^2 - ε))
		# u = √v
		#		(v = (σ^2 - ε))
		dt = np.sum(dxn * self.xc, axis=0)
			# dt = (x-μ) * Σdx^
		du = - dt / (self.std * self.std)
			# du = dt * (- 1 / u^2)
		dvar = du * 0.5 / self.std
			# dv = du * 0.5 / √v
		#dvar = dstd * 0.5 / self.std
		#dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
				# dσ = d√(σ^2 - ε) * (0.5 / √(σ^2 - ε))
		
		# σ^2 = 1/N * Σ(x-μ)^2
		dxc += dvar * (2.0 / self.batch_size) * self.xc
				# d(x-μ) = dx^ / √(σ^2 - ε) + 
				#          dvar * 1/N * 2 * (x-μ)
		# x-μ
		dx1 = dxc					# dx1 = dxc * 1
		dmu = -np.sum(dxc, axis=0)	# dμ = Σdxc * (-1)
		
		# μ = 1/N * Σx
		dx2 = dmu / self.batch_size	# dμ = dμ * 1/N * 1
		
		dx = dx1 + dx2
		
		# dγ, dβをクラス変数に設定
		self.dgamma = dgamma
		self.dbeta  = dbeta
		
		return dx
	
#################################################
# ドロップアウト
##################################################
class Dropout:
	
	# コンストラクタ
	def __init__(self, drouput_ratio=0.5):
		self.dropout_ratio = dropout_ratio
		self.mask = None
	
	# 順伝播
	def forward(self, x, train_flg=True):
		
		if self.train_flg:
			self.mask = np.random.rand(*x.shape) > self.dropout_ratio
			return x * self.mask
		else:
			return x * (1.0 - self.drouput_ratio)
		
	# 逆伝播
	def backward(self, dout=1):
		
		return dout * self.mask
		
##################################################
# 畳み込み層
##################################################
class Convolution:
	
	# コンストラクタ
	def __init__(self, W, b, stride=1, padding=0):
		self.W = W
		self.b = b
		self.stride = stride
		self.padding = padding
		
		# 中間データ(backward時に使用)
		self.x = None
		self.col_img = None
		self.col_W = None
		
		# 重み・バイアスパラメータの勾配
		self.dW = None
		self.db = None
		
	# 順伝播
	def forward(self, x):
		
		FN, C, FH, FW = self.W.shape
		N, C, H, W = x.shape
		out_h = int( (H + 2 * self.padding - FH) / self.stride ) + 1
		out_w = int( (W + 2 * self.padding - FW) / self.stride ) + 1
		
		# 2次元の配列に変換し、行列演算しやすくする
		col_img = im2col(x, FH, FW, self.stride, self.padding)
		col_W = self.W.reshape(FN, -1).T
		
		# 畳み込み演算
		out = np.dot(col_img, col_W) + self.b
			# (N*?, C*FH*FW) × (FN, C*FH*FW).T
		out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
			# 0->0：N(データ数)
			# 3->1：FN(フィルタ数)
			# 1->2：out_h(出力高さ)
			# 2->3：out_w(出力幅)
		
		# backward時に使用するので保持する
		self.x = x
		self.col_img = col_img
		self.col_W = col_W
		
		return out
		
	# 逆伝播
	def backward(self, dout):
		
		FN, C, FH, FW = self.W.shape
		dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
			# 0->0：N(データ数)
			# 2->1：out_h(出力高さ)
			# 3->2：out_w(出力幅)
			# 1->3：FN(フィルタ数)
		
		self.db = np.sum(dout, axis=0)
		self.dW = np.dot(self.col_img.T, dout)
		self.dW = self.dW.transpose(1,0).reshape(FN, C, FH, FW)
			# 1->0：FN
			# 0->1：C * FH * FW
		
		dcol = np.dot(dout, self.col_W.T)
			# (N*out_h*out_w, FN).T × (FN, C*FH*FW)
			#		-> (C*FH*FW, N*out_h*out_w)
		dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.padding)
			# -> (N,C,H,W)
		
		return dx
	
#################################################
# プーリング層
##################################################
class Pooling:
	
	# コンストラクタ
	def __init__(self, pool_h, pool_w, stride=1, padding=0):
		self.pool_h = pool_h
		self.pool_w = pool_w
		self.stride = stride
		self.padding = padding
		self.pool_size = pool_h * pool_w
		
		# 中間データ(backward時に使用)
		self.x = None
		self.arg_max = None
	
	# 順伝播
	def forward(self, x, train_flg=True):
		
		N, C, H, W = x.shape
		out_h = int( (H - self.pool_h) / self.stride ) + 1
		out_w = int( (W - self.pool_w) / self.stride ) + 1
		
		# 2次元の配列に変換し、行列演算しやすくする
		col_img = im2col(x, self.pool_h, self.pool_w, self.stride, self.padding)
			# -> (N*out_h*out_w, C*pool_h*pool_w)
		col_img = col_img.reshape(-1, self.pool_size)
			# -> (N*C*out_h*out_w, pool_h*pool_w)
		
		# 最大値を取った位置(インデックス)を保持する
		arg_max = np.argmax(col_img, axis=1)
		
		# (pool_x × pool_h)から最大値を取得する
		out = np.max(col_img, axis=1)
		out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
			# 0->0：N(データ数)
			# 3->1：C(チャネル数)
			# 1->2：out_h(出力高さ)
			# 2->3：out_w(出力幅)
		
		# 中間データを保存する
		self.x = x
		self.arg_max = arg_max
		
		return out
		
	# 逆伝播
	def backward(self, dout=1):
		
		dout = dout.transpose(0, 2, 3, 1)
			# 0->0：N(データ数)
			# 2->1：out_h(出力高さ)
			# 3->2：out_w(出力幅)
			# 1->3：C(チャネル数)
		
		# プーリング層では順伝播のときに
		# 最大値を取得した画素に逆伝播させる
		dmax = np.zeros((dout.size, self.pool_size))
				# shape=(N*out_h*out_w*C, pool_h*pool_w)
		seq_i = np.arange(self.arg_max.size)
		arg_max_idx = self.arg_max.flatten()
		dmax[seq_i, arg_max_idx] = dout.flatten()
				# shape=(N*out_h*out_w*C, pool_h*pool_w)
		
		# 最大値以外の画素要素も確保する
		dmax = dmax.reshape(dout.shape + (self.pool_size,))
			# shape=(N, out_h, out_w, C, pool_h*pool_w)
			
		# 入力のサイズに戻す
		dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
				# shape=(N*out_h*out_w, C*pool_h*pool_w)
		dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.padding)
				# shape=(N, C, H, W)
		
		return dx
		
