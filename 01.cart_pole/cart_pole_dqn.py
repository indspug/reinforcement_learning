# -*- coding: utf-8 -*-
#
##################################################
# OpenAI GymのCartPoleをQ-Learningで強化学習する
##################################################

import gym
import numpy as np
import sys, os
sys.path.append(os.pardir)
from multi_layer_net import MultiLayerNet
from common.optimizer import *
#import matplotlib.pylab as plt


# パラメータ
NUM_EPISODES = 100000	# 教師あり学習間に学習するエピソード数
MAX_STEP = 200			# 1エピソードの最大ステップ

STATE_NUM = 4

EXPERIENCE_MEM_SIZE = 1000

BATCH_SIZE = 32

# 状態・行動を離散化する際の分割数
CART_POS_BIN   = 4
CART_V_BIN     = 4
POLE_ANGLE_BIN = 4
POLE_V_BIN     = 4
ACTION_BIN     = 2

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
def get_action(network, sequence, episode):
	
	# ε-greedy(epsilon-greedy)法で次のアクション取得
	epsilon = 0.01 + 0.9 / (1.0 + episode)
	#epsilon = 1.0 * (0.999 ** episode)
	#if epsilon < 0.01:
	#	epsilon = 0.01
	
	sequence = sequence[np.newaxis,:]
	if epsilon <= np.random.uniform(0, 1):
		actions = network.predict(sequence)
		next_action = np.argmax(actions)
	else:
		next_action = np.random.choice([0, 1])
	
	return next_action
	
##################################################
# 状態(state)を更新する
##################################################
def renew_sequence(sequence, new_state, state_num):

	sequence = np.r_[new_state, sequence]
	sequence = np.delete(sequence, range(STATE_NUM*(STATE_MEM_SIZE-1), STATE_NUM*STATE_MEM_SIZE))
	#sequence = np.hstack(sequence, new_state)
	#sequence = np.r_[sequence, new_state]
	return sequence
	
##################################################
# メイン
##################################################
if __name__ == '__main__':
	
	# 環境読み込み
	env = gym.make('CartPole-v0')
	
	# Experience Memory初期化
	experience_memory = []

	# Neural Network初期化
	network = MultiLayerNet(input_size=STATE_NUM, hidden_size_list=[16,16], \
							output_size=ACTION_BIN)
	optimizer = Adam(lr=0.01)
	#optimizer = SGD(lr=0.001)
		
	ave_steps = 0
	ave_loss = 0.0

	# 強化学習開始
	for episode in range(NUM_EPISODES):

		# 環境初期化
		observation = env.reset()
		state = np.array(observation)
		experience_memory_local = []
		
		# 最大ステップ数まで学習
		for step in range(MAX_STEP):
			
			# 次の行動を決める
			action = get_action(network, state, episode)
			#if(episode == 0):
			#	print('step %03d : cart(pos,v)=(%.2f,%.2f) pole(ang,v)=(%.2f,%.2f) action:%d' %
			#		(step+1, observation[0], observation[1], 
			#			observation[2], observation[3],
			#			action)
			#		)

			# 一定エピソード毎に1回描画
			if (episode % 1000) == 0:
				env.render()	# 描画
			
			# 1ステップ進める
			observation, reward, done, info = env.step(action)
				# 取得したアクション後の状態,報酬,終了判定,情報
			new_state = np.array(observation)
			#new_state = digitize_state(observation)
			#sequence = renew_sequence(sequence, new_state, STATE_NUM)
			#new_sequence = sequence.copy()

			# 終了(ポールが倒れた)場合は大きい罰則を与える
			if done:
				#new_state = np.zeros(STATE_NUM)
				if step < 195:
					reward = -200
				else:
					reward = 1
			else:
				reward = 1

			# エピソードローカルなメモリに保存
			#new_experience = np.r_[old_sequence, np.array(action, reward)]
			#new_experience = no.r_[new_experience, new_sequence]
			experience_memory_local.append(
				np.r_[state, np.array([float(action), reward]), new_state])

			# 終了(ポールが倒れた)場合
			if done:
				ave_steps += step
				#print('Trial-%03d Episode-%03d finished at %03d steps' % (trial+1, episode+1, step+1))
				break

			# 状態を更新する
			state = new_state.copy()

		#
		# エピソードローカルなメモリ -> グローバルなメモリに移す
		for experience in experience_memory_local:
			experience_memory.append(experience)
		if len(experience_memory) >= EXPERIENCE_MEM_SIZE:
			delete_size = len(experience_memory) - EXPERIENCE_MEM_SIZE
			experience_memory = experience_memory[delete_size:]

		# Neural Networkの更新
		len_experience_memory = len(experience_memory)
		if len_experience_memory >= BATCH_SIZE:
			# バッチデータ取り出し
			batch_idx = np.random.choice(len_experience_memory, BATCH_SIZE)
			batch_ex = np.array( [experience_memory[i] for i in batch_idx])
			x = batch_ex[:,0:STATE_NUM].reshape(BATCH_SIZE,-1)
			target = network.predict(x)
			
			# 教師データ作成
			for i in range(BATCH_SIZE):
				action = int(batch_ex[i, STATE_NUM])
				reward = batch_ex[i, STATE_NUM + 1]
				next_state = batch_ex[i, STATE_NUM + 2:]
				next_actions = network.predict(next_state[np.newaxis,:])
				#print(next_actions)
				target[i,action] = reward + GAMMA * np.max(next_actions)
				
			# ネットワーク更新
			grads = network.gradient(x, target)
			optimizer.update(network.params, grads)

			loss = network.loss(x, target)
			ave_loss += loss

		# 一定エピソード毎に損失と終了するまでの平均ステップを表示する
		if (episode % 100) == 0:
			ave_loss /= float(100)
			ave_steps /= 100
			print('Episode-%07d  average steps:%03d  average loss:%0.3f' % (episode+1, ave_steps, ave_loss))
			ave_loss = 0
			ave_steps = 0
	#
	
