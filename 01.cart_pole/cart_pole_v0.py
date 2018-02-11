# -*- coding: utf-8 -*-
#
##################################################
# OpenAI GymのCartPoleをQ-Learningで強化学習する
##################################################

import gym
import numpy as np
import matplotlib.pylab as plt

# パラメータ
NUM_EPISODES = 5000
MAX_STEP = 200

CART_POS_BIN   = 4
CART_V_BIN     = 4
POLE_ANGLE_BIN = 4
POLE_V_BIN     = 4

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
	cart_pos, cart_v, pole_angle, pole_v = observation
			# カート位置,カート速度, ポールの角度, ポールの速度
	
	# ビンの位置(インデックス)を返す
	digitized = [	np.digitize(cart_pos,	bins(-2.4, 2.4, CART_POS_BIN)	),
					np.digitize(cart_v,		bins(-3.0, 3.0, CART_V_BIN)		),
					np.digitize(pole_angle,	bins(-0.5, 0.5, POLE_ANGLE_BIN)	),
					np.digitize(pole_v,		bins(-3.0, 3.0, POLE_V_BIN)		)	]
	
	return digitized

##################################################
# 次のアクションを返す
##################################################
#def get_action(q_table, state, action, observation, reward):
def get_action(q_table, state, action, observation, reward, episode):
	next_state = digitize_state(observation)		# 次の状態
	
	# ε-greedy(epsilon-greedy)法で次のアクション取得
	#epsilon = 0.2
	epsilon = 0.5 * (0.999 ** episode)
	if epsilon <= np.random.uniform(0, 1):
		next_action = np.argmax(q_table[next_state])	# 次の状態から最善のアクションを取得
	else:
		next_action = np.random.choice([0, 1])
	
	i1 = state + [action]
	i2 = next_state + [next_action]
	q_table[i1] = (1.0 - ALPHA) * q_table[i1] + \
					ALPHA * (reward + GAMMA * q_table[i2])
	#print(next_action, next_state)
	
	return next_action, next_state
	
##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 環境読み込み
	env = gym.make('CartPole-v0')
	
	# Q関数の初期化
	#   sizeは(カート位置,カート速度, ポールの角度, ポールの速度, アクション)
	q_table = np.random.uniform( \
					low=-1, high=1, \
					size=(CART_POS_BIN, CART_V_BIN, POLE_ANGLE_BIN, POLE_V_BIN, 2) )
	
	# 学習の履歴
	steps_history = []
	
	# 強化学習開始
	for episode in range(NUM_EPISODES):
		
		# 環境初期化
		observation = env.reset()
		state = digitize_state(observation)
		action = np.argmax(q_table[state])
		
		# 最大ステップ数まで学習
		for step in range(MAX_STEP):
			
			#if episode >= NUM_EPISODES-10:
			#	env.render()	# 描画
			
			# 1ステップ進める
			observation, reward, done, info = env.step(action)
				# 取得したアクション後の状態,報酬,終了判定,情報
			
			# 終了(ポールが倒れた)場合は大きい罰則を与える
			if done:
				reward = -200
				
			#action = env.action_space.sample()	# ランダムなアクション取得
			#action, state = get_action(q_table, state, action, observation, reward)
			action, state = get_action(q_table, state, action, observation, reward, episode)
			
			# 終了(ポールが倒れた)場合
			if done:
				steps_history.append(step+1)
				print('Episode-%04d finished at %03d steps' % (episode, step))
				break
		
	#episodes = np.arange(len(steps_history))
	#plt.plot(episodes, steps_history)
	episode_list = []
	step_list = []
	ave = 0
	for i, step in enumerate(steps_history):
		ave += step
		if (i+1) % 50 == 0:
			episode_list.append(i+1)
			step_list.append(ave/50)
			ave = 0
	plt.plot(episode_list, step_list)
	plt.xlabel('Episodes')
	plt.ylabel('Steps')
	plt.show()
	
