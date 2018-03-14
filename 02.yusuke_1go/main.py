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
import matplotlib.pylab as plt

# パラメータ
NUM_EPISODES = 100000
MAX_STEP = 500

CART_X_BIN     = 10
CART_Y_BIN     = 5
CART_ANGLE_BIN = 20
RSPEED_BIN     = 10

ALPHA = 0.2		# 学習率
GAMMA = 0.99	# 割引率

##################################################
# ビン(min〜maxをnum分割した値)を返す
##################################################
def bins(clip_min, clip_max, num):
	bins = np.linspace(clip_min, clip_max, num+1)[1:-1]
	return bins
	
##################################################
# 状態を離散値に変換する
##################################################
def digitize_state(observation):
	cart_x, cart_y, cart_angle = observation
			# カートX位置, カートY位置, カートの角度,
	
	# ビンの位置(インデックス)を返す
	digitized = [	np.digitize(cart_x,	    bins(-100,  100, CART_X_BIN)	    ),
					np.digitize(cart_y,		bins(   0, 1000, CART_Y_BIN)		),
					np.digitize(cart_angle,	bins(-math.pi, math.pi, CART_ANGLE_BIN)	)
				]
	
	return digitized

##################################################
# アクションを離散値に変換する
##################################################
def digitize_action(action):
	left_rspeed, right_rspeed = action
			# 左ホイールの速度, 右ホイールの速度
	
	# ビンの位置(インデックス)を返す
	digitized = [	np.digitize(left_rspeed,  bins(0, 10, RSPEED_BIN) ),
					np.digitize(right_rspeed, bins(0, 10, RSPEED_BIN) )
				]
	
	return digitized


##################################################
# 次のアクションを返す
##################################################
#def get_action(q_table, state, action, observation, reward):
def get_action(q_table, state, action, observation, reward, episode):
	next_state = digitize_state(observation)		# 次の状態
	action_bin = digitize_action(action)
	
	# ε-greedy(epsilon-greedy)法で次のアクション取得
	#epsilon = 0.2
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
	#print(next_action, next_state)

	return next_action, next_state
	
##################################################
# 最善のアクションのインデックスを返す
##################################################
def get_action_argmax(q_table, state):

	new_actions = q_table[state]

	qmax = -1e5
	max_i, max_j = -1, -1
	for i in range(new_actions.shape[0]):
		for j in range(new_actions.shape[1]):
			if new_actions[i, j] > qmax:
				qmax = new_actions[i, j] 
				max_i = i
				max_j = j

	return max_i, max_j
	
##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 環境読み込み
	#env = gym.make('CartPole-v0')
	env = Yusuke1goEnv()
	
	# Q関数の初期化
	#   sizeは(カートX位置,カートY位置, カートの角度, 左ホイールの速度、右ホイールの速度)
	q_table = np.random.uniform( \
					low=-1, high=1, \
					size=(CART_X_BIN, CART_Y_BIN, CART_ANGLE_BIN, RSPEED_BIN, RSPEED_BIN) )
	
	# 学習の履歴
	steps_history = []
	
	# 強化学習開始
	for episode in range(NUM_EPISODES):
		
		# 環境初期化
		observation = env.reset()
		state = digitize_state(observation)
		#action = np.argmax(q_table[state])
		action = get_action_argmax(q_table, state)
		
		# 最大ステップ数まで学習
		for step in range(MAX_STEP):
			
			if (episode % 100) == 0:
				env.render()	# 描画
			
			lspeed = float(action[0]) / float(RSPEED_BIN) * 10
			rspeed = float(action[1]) / float(RSPEED_BIN) * 10
			
			# 1ステップ進める
			observation, reward, done, info = env.step((lspeed, rspeed))
			#observation, reward, done, info = env.step(action)
				# 取得したアクション後の状態,報酬,終了判定,情報
			
			#print('x=%f, y=%f, dir=%f' % (observation[0], observation[1], observation[2]))

			# 終了(ポールが倒れた)場合は大きい罰則を与える
			#if done:
				#reward = -200
				
			action, state = get_action(q_table, state, action, observation, reward, episode)
			
			# 終了(ポールが倒れた)場合
			if done:
				steps_history.append(step+1)
				print('Episode-%04d finished at %03d steps' % (episode, step))
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
	
