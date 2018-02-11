# coding: utf-8

import numpy as np

##################################################
# 画像データを2次元配列に変換する
##################################################
def im2col(input_data, filter_h, filter_w, stride=1, padding=0):
	
	# N：データ数、C：チャネル数、H：高さ、W：幅
	N, C, H, W = input_data.shape
	
	# 2次元配列のH × W (//は小数点以下切捨ての除算)
	out_h = (H + 2*padding - filter_h) // stride + 1
	out_w = (W + 2*padding - filter_w) // stride + 1
	
	# Nはパディング無し
	# Cはパディング無し
	# Hの先頭と末尾にpadding要素パディング
	# Wの先頭と末尾にpadding要素パディング
	img = np.pad(	input_data, \
					[(0,0), (0,0), (padding,padding), (padding, padding)], \
					'constant')
	
	# col：返す2次元配列
	col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
	for y in range(filter_h):
		y_end = y + stride * out_h
		for x in range(filter_w):
			x_end = x + stride * out_w
			col[:, :, y, x, :, :] = \
			img[:, :, y:y_end:stride, x:x_end:stride]
				# col.shape=(N,C,1,1,out_h,out_w)
				# img.shape=(N,C,    out_h,out_w)
				# コピー先(col)のy,xは固定し、
				# y〜(y_end), x〜(x_end)までズラしながら取り出す
	
	col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
		# 0：N(データ数)
		# 4：Y方向計算回数(out_h)
		# 5：X方向計算回数(out_w)
		# 1：C(チャネル数)
		# 2：filter_h(フィルタ高さ)
		# 3：filter_w(フィルタ幅)
	
	return(col)
	
##################################################
# 2次元配列を画像データに変換する
##################################################
def col2im(col, input_shape, filter_h, filter_w, stride=1, padding=0):
	
	# N：データ数、C：チャネル数、H：高さ、W：幅
	N, C, H, W = input_shape
	
	# 2次元配列のH × W (//は小数点以下切捨ての除算)
	out_h = (H + 2*padding - filter_h) // stride + 1
	out_w = (W + 2*padding - filter_w) // stride + 1
	
	#
	col2 = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0,3,4,5,1,2)
		# 0->0：N(データ数)
		# 3->1：C(チャネル数)
		# 4->2：filter_h(フィルタ高さ)
		# 5->3：filter_w(フィルタ幅)
		# 1->4：Y方向計算回数(out_h)
		# 2->5：X方向計算回数(out_w)
	
	# img：返す画像データ
	img = np.zeros((N, C, \
					H + 2*padding + stride - 1, \
					W + 2*padding + stride - 1))
	
	for y in range(filter_h):
		y_end = y + stride * out_h
		for x in range(filter_w):
			x_end = x + stride * out_w
			img[:, :, y:y_end:stride, x:x_end:stride] += \
			col2[:, :, y, x, :, :] 
				# col.shape=(N,C,1,1,out_h,out_w)
				# img.shape=(N,C,    out_h,out_w)
				# コピー先(col)のy,xは固定し、
				# y〜(y_end), x〜(x_end)までズラしながら取り出す
	
	img2 = img[:, :, padding:H+padding, padding:W+padding]
	return(img2)
	
