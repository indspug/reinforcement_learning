# -*- coding: utf-8 -*-
#
##################################################
# OpenAI Gymの自作環境「yusuke_1go」をQ-Learningで強化学習する
##################################################

import sys, os
import myenv
import math
import numpy as np
import gym
from gym import wrappers  # gymの画像保存
import time, datetime
import pickle

# パラメータ
NUM_EPISODES = 10000000
MAX_STEP = 1000

MOVIE_SAVE_CYCLE = 10000	# ムービー保存周期(エピソード単位)
RENDER_CYCLE = 1000		# 描画周期(エピソード単位)

CART_X_BIN     = 4	# Xの離散化数
CART_Y_BIN     = 2	# Yの離散化数
CART_ANGLE_BIN = 16	# 角度の離散化数
SENSOR_BIN     = 8	# センサ値の離散化数
RSPEED_BIN     = 4	# 回転速度の離散化数

#ALPHA = 0.2		# 学習率
#GAMMA = 0.7		# 割引率
ALPHA = 0.1		# 学習率
GAMMA = 0.9		# 割引率

RESULT_OUTPUT_CYCLE = 1000	# 結果出力サイクル(エピソード単位)
RESULT_CSV = "./result.csv"	# 結果ファイル

QVALUE_SAVE_CYCLE = 50000	# Q値保存周期(エピソード単位)
QVALUE_SAVE_DIR = "./q_value"	# 値保存ディレクトリ
 
##################################################
# 連続値を離散化した(min〜maxをnum分割した値)を返す
##################################################
def bins(clip_min, clip_max, num):
	bins = np.linspace(clip_min, clip_max, num+1)[1:-1]
	return bins
	
##################################################
# 状態を離散値に変換する
##################################################
def digitize_state(observation, observation_space):

	# カートX位置, カートY位置, カートの角度, センサー値
	cart_x, cart_y, cart_angle, sensor1, sensor2, sensor3, sensor4, sensor5 = observation
	
	# 最大値と最小値
	cart_x_high, cart_y_high, cart_angle_high, \
		sensor1_high, sensor2_high, sensor3_high, sensor4_high, sensor5_high = observation_space.high
	cart_x_low,  cart_y_low,  cart_angle_low,  \
		sensor1_low,  sensor2_low,  sensor3_low, sensor4_low, sensor5_low  = observation_space.low
	
	# ビンの位置(インデックス)を返す
	digitized = [	np.digitize(cart_angle,	bins(cart_angle_low, cart_angle_high, CART_ANGLE_BIN)),
					np.digitize(sensor1,	bins(sensor1_low,    sensor1_high,    SENSOR_BIN)    ),
					np.digitize(sensor2,	bins(sensor2_low,    sensor2_high,    SENSOR_BIN)    ),
					np.digitize(sensor3,	bins(sensor3_low,    sensor3_high,    SENSOR_BIN)    ),
					np.digitize(sensor4,	bins(sensor4_low,    sensor4_high,    SENSOR_BIN)    ),
					np.digitize(sensor5,	bins(sensor5_low,    sensor5_high,    SENSOR_BIN)    )
				]
	
	return digitized

##################################################
# アクションを離散値に変換する
##################################################
def digitize_action(action, action_space):

	# 左ホイールの速度, 右ホイールの速度
	left_rspeed, right_rspeed = action
	
	# 最大値と最小値
	left_rs_high, right_rs_high = action_space.high
	left_rs_low,  right_rs_low  = action_space.low

	# ビンの位置(インデックス)を返す
	digitized = [	np.digitize(left_rspeed,  bins(left_rs_low,  left_rs_high,  RSPEED_BIN) ),
					np.digitize(right_rspeed, bins(right_rs_low, right_rs_high, RSPEED_BIN) )
				]
	
	return digitized

##################################################
# 最善のアクションのインデックスを返す
##################################################
def get_action_argmax(q_table, state):

	new_actions = q_table[state]

	qmax = -1e300
	max_i, max_j = -1, -1
	for i in range(new_actions.shape[0]):
		for j in range(new_actions.shape[1]):
			if new_actions[i, j] > qmax:
				qmax = new_actions[i, j] 
				max_i = i
				max_j = j

	return np.array([max_i, max_j])
	
##################################################
# アクションを離散値から連続値に変換する
##################################################
def d2a_action_argmax(action, action_space):

	# 左ホイールの速度, 右ホイールの速度
	left_rspeed_bin, right_rspeed_bin = action
	
	# 最大値と最小値
	left_rs_high, right_rs_high = action_space.high
	left_rs_low,  right_rs_low  = action_space.low

	# 範囲
	left_range  = left_rs_high  - left_rs_low
	right_range = right_rs_high - right_rs_low

	# 連続値計算
	lspeed = float(left_rspeed_bin)  / float(RSPEED_BIN-1) * left_range  + left_rs_low
	rspeed = float(right_rspeed_bin) / float(RSPEED_BIN-1) * right_range + right_rs_low

	return np.array([lspeed, rspeed])

##################################################
# 次のアクションを返す
##################################################
def get_action(	q_table, state, 
				action, action_space, 
				observation, observation_space, 
				reward, episode):

	next_state = digitize_state(observation, observation_space)
	action_bin = digitize_action(action, action_space)
	
	# ε-greedy(epsilon-greedy)法で次のアクション取得
	epsilon = 0.5 * (0.999 ** episode)
	if epsilon <= np.random.uniform(0, 1):
		# 次の状態から最善のアクションを取得
		next_action = get_action_argmax(q_table, next_state)
	else:
		next_action = np.random.randint(0, RSPEED_BIN, 2)
	
	i1 = state + [action_bin[0], action_bin[1]]
	i2 = next_state + [next_action[0], next_action[1]]
	q_table[i1] = (1.0 - ALPHA) * q_table[i1] + \
					ALPHA * (reward + GAMMA * q_table[i2])

	return next_action, next_state

##################################################
# 日付の文字列表現を返す
##################################################
def get_datetime_string():
	
	dn = datetime.datetime.now()
	return "%04d/%02d/%02d %02d:%02d:%02d" % (
		dn.year, dn.month, dn.day,
		dn.hour, dn.minute, dn.second	
	)
	
##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# [usage] python3 main.py [pickle_path] [episode]
	#   
	#   pickle_path:保存したQ値のファイルパスを指定
	#   episode    :学習を再開するエピソードを指定(整数)
	#
	# ex) python3 main.py ./q_value/q_value_0001000.pickle 1001

	# コマンドライン引数取得
	argvs = sys.argv
	argc = len(argvs)
	start_episode = 0
	pickle_path = ""
	if (argc >= 3):
		pickle_path = argvs[1]		# 保存したQ値を取得
		start_episode = int(argvs[2])	# 開始エピソードを取得

	# 環境読み込み
	env = gym.make('myenv-v0')
	env = wrappers.Monitor(
		env, './movie', force=True, 
		video_callable=(lambda ep: (ep+start_episode) % MOVIE_SAVE_CYCLE == 0))

	observation_space = env.observation_space
	action_space = env.action_space

	# 保存したQ値読み込み
	if pickle_path:
		with open (pickle_path, 'rb') as fin:
			q_table = pickle.load(fin)
	else:
		# Q関数の初期化
		#   sizeは(カートX位置,カートY位置, カートの角度, センサー値, 左ホイールの速度、右ホイールの速度)
		q_table = np.random.uniform( \
				low=-1, high=1, \
				#size=(	CART_X_BIN, CART_Y_BIN, CART_ANGLE_BIN, 
				#		SENSOR_BIN, SENSOR_BIN, SENSOR_BIN, SENSOR_BIN, SENSOR_BIN,
				#		RSPEED_BIN, RSPEED_BIN) )
				size=(	
					CART_ANGLE_BIN, 
					SENSOR_BIN, SENSOR_BIN, SENSOR_BIN, SENSOR_BIN, SENSOR_BIN, 
					RSPEED_BIN, RSPEED_BIN
				)
			)

		
	
	# 結果ファイル生成
	fo = open(RESULT_CSV, "a")
	fo.write("cart_x_bin,%d\n"     % (CART_X_BIN))
	fo.write("cart_y_bin,%d\n"     % (CART_Y_BIN))
	fo.write("cart_angle_bin,%d\n" % (CART_ANGLE_BIN))
	fo.write("sensor_bin,%d\n"     % (SENSOR_BIN))
	fo.write("rspeed_bin,%d\n"     % (RSPEED_BIN))
	fo.write("alpha,%f\n"          % (ALPHA))
	fo.write("gamma,%f\n"          % (GAMMA))
	fo.write("episode,goal_num,average_y,average_step,elapsed_time[sec]\n")
	fo.close()

	# 学習の履歴
	goal_num = 0
	average_y = 0
	average_step = 0
	start_time = time.time()
	
	# 強化学習開始
	for episode in range(start_episode, NUM_EPISODES):
		
		# 環境初期化
		observation = env.reset()
		state = digitize_state(observation, observation_space)
		action_bin = get_action_argmax(q_table, state)
		
		# 描画有無決定
		isRendering = False
		if (episode % RENDER_CYCLE) == 0:
			isRendering = True

		# 最大ステップ数まで学習
		for step in range(MAX_STEP):
			
			# 描画
			if isRendering:
				env.render()
			
			# アクションを離散値から連続値に変換
			action = d2a_action_argmax(action_bin, action_space)

			# 1ステップ進める
			observation, reward, done, info = env.step(action)
				# 取得したアクション後の状態,報酬,終了判定,情報
			
			# (離散化された)次のアクションと離散化した状態を取得
			action_bin, state = get_action(	q_table, state, action, action_space,
										observation, observation_space, 
										reward, episode)
			
			# 終了の場合
			if done:
				y = observation[1]
				average_y = average_y + y
				average_step = average_step + (step+1)
				ymax = observation_space.high[1]
				if (y >= ymax):
					goal_num = goal_num + 1
				print('[%s] %08d,%03d,%d' % (get_datetime_string(), episode, step, y))
				break
			
		
		# Q値の保存
		if (episode % QVALUE_SAVE_CYCLE) == 0:
			if not os.path.exists(QVALUE_SAVE_DIR):
				os.makedirs(QVALUE_SAVE_DIR)
			pickle_path = "{0}/{1}_{2:07d}.pickle".format(
				QVALUE_SAVE_DIR, 'q_value', episode)
			with open(pickle_path, 'wb') as fout:
				pickle.dump(q_table, fout)

		# 結果出力
		if (episode % RESULT_OUTPUT_CYCLE) == 0:
			average_y = average_y / RESULT_OUTPUT_CYCLE
			average_step = average_step / RESULT_OUTPUT_CYCLE
			elapsed_time = time.time() - start_time
			fo = open(RESULT_CSV, "a")
			fo.write("%d, %d, %f, %f, %f\n" % (episode, goal_num, average_y, average_step, elapsed_time))
			fo.close()
			goal_num = 0
			average_y = 0
			average_step = 0

	# 環境終了
	env.close()

