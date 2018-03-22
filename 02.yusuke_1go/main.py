# -*- coding: utf-8 -*-
#
##################################################
# OpenAI Gymの自作環境「yusuke_1go」をQ-Learningで強化学習する
##################################################

import sys, os
sys.path.append(os.getcwd())
from yusuke_1go import Yusuke1goEnv
import math
import numpy as np
#import matplotlib.pylab as plt

# パラメータ
NUM_EPISODES = 10000000
MAX_STEP = 1000

CART_X_BIN     = 4	# Xの離散化数
CART_Y_BIN     = 2	# Yの離散化数
CART_ANGLE_BIN = 16	# 角度の離散化数
SENSOR_BIN     = 4	# センサ値の離散化数
RSPEED_BIN     = 8	# 回転速度の離散化数

ALPHA = 0.2		# 学習率
GAMMA = 0.7	# 割引率

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
	#digitized = [	np.digitize(cart_x,	    bins(cart_x_low,     cart_x_high,     CART_X_BIN)    ),
	#				np.digitize(cart_y,		bins(cart_y_low,     cart_y_high,     CART_Y_BIN)    ),
	#				np.digitize(cart_angle,	bins(cart_angle_low, cart_angle_high, CART_ANGLE_BIN)),
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
	#print(new_actions)

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

	#new_state = digitize_state(observation, observation_space)
	#next_state = new_state + state
	#del next_state[STATE_NUM*(STATE_TIME_NUM-1) : STATE_NUM*STATE_TIME_NUM]
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
# メイン
##################################################
if __name__ == '__main__':
	
	# 環境読み込み
	#env = gym.make('CartPole-v0')
	env = Yusuke1goEnv(MAX_STEP)
	observation_space = env.observation_space
	action_space = env.action_space

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
	
	# 学習の履歴
	#steps_history = []
	ave_step = 0
	ave_reward = 0
	
	# 強化学習開始
	for episode in range(NUM_EPISODES):
		
		# 環境初期化
		observation = env.reset()
		state = digitize_state(observation, observation_space)
		#state = state + state + state
		action_bin = get_action_argmax(q_table, state)
		
		# 描画有無決定
		isRendering = False
		if (episode > 20000) and (episode % 200) == 0:
			isRendering = True
		elif (episode > 5000) and (episode % 500) == 0:
			isRendering = True
		elif (episode > 500) and (episode % 1000) == 0:
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
			#print('x=%f, y=%f, dir=%f' % (observation[0], observation[1], observation[2]))
			ave_reward = ave_reward + reward
			
			# (離散化された)次のアクションと離散化した状態を取得
			action_bin, state = get_action(	q_table, state, action, action_space,
										observation, observation_space, 
										reward, episode)
			
			# 終了の場合
			if done:
				ave_step = ave_step + (step + 1)
				#steps_history.append(step+1)
				#print('Episode-%08d finished at %03d steps' % (episode, step))
				if (episode % 500) == 0:
					ave_step = ave_step / 500
					ave_reward = ave_reward / 500
				    #print('Episode-%08d finished at %03d steps' % (episode, step))
					print('Episode-%08d finished at average %03d steps, average rewards %7.1f' 
							% (episode, ave_step, ave_reward))
					ave_step = 0
					ave_reward = 0

				break
		
	#episodes = np.arange(len(steps_history))
	#plt.plot(episodes, steps_history)

	# 環境終了
	env.close()

	#episode_list = []
	#step_list = []
	#ave = 0
	#for i, step in enumerate(steps_history):
	#	ave += step
	#	if (i+1) % 50 == 0:
	#		episode_list.append(i+1)
	#		step_list.append(ave/50)
	#		ave = 0
	#plt.plot(episode_list, step_list)
	#plt.xlabel('Episodes')
	#plt.ylabel('Steps')
	#plt.show()

