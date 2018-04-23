# -*- coding: utf-8 -*-
#
##################################################
# OpenAI Gymの自作環境「yusuke_1go」をDQNで強化学習する
##################################################

import sys, os
sys.path.append(os.getcwd())
from yusuke_1go import Yusuke1goEnv
import math
import numpy as np
#import matplotlib.pylab as plt
from gym import wrappers  # gymの画像保存
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
from collections import deque


# パラメータ
NUM_EPISODES = 10000000
MAX_STEP = 1000

CART_X_BIN     = 4	# Xの離散化数
CART_Y_BIN     = 2	# Yの離散化数
CART_ANGLE_BIN = 16	# 角度の離散化数
SENSOR_BIN     = 4	# センサ値の離散化数
#RSPEED_BIN     = 8	# 回転速度の離散化数
RSPEED_BIN     = 4	# 回転速度の離散化数

ALPHA = 0.02		# 学習率
GAMMA = 0.7	# 割引率

DQN_MODE = 1    # 1がDQN、0がDDQN
num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
total_reward_vec = np.zeros(num_consecutive_iterations)  # 各試行の報酬を格納

hidden_size = 16               # Q-networkの隠れ層のニューロンの数
#learning_rate = 0.00001         # Q-networkの学習係数
learning_rate = 0.01         # Q-networkの学習係数
memory_size = 10000            # バッファーメモリの大きさ
#batch_size = 32                # Q-networkを更新するバッチの大記載
batch_size = 1                # Q-networkを更新するバッチの大記載
 

# [1]損失関数の定義
# 損失関数に関数を使用します
def huberloss(y_true, y_pred):
    return K.mean(K.minimum(0.5*K.square(y_pred-y_true), K.abs(y_pred-y_true)-0.5), axis=1)

# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
#    def __init__(self, learning_rate=0.01, state_size=4, action_size=2, hidden_size=10):
#    def __init__(self, learning_rate=0.01, state_size=8, action_size=64, hidden_size=10):
    def __init__(self, learning_rate=0.01, state_size=6, action_size=RSPEED_BIN*RSPEED_BIN, hidden_size=10):
        self.model = Sequential()
        self.model.add(Dense(hidden_size, activation='relu', input_dim=state_size))
        self.model.add(Dense(hidden_size, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))
        self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
        # self.model.compile(loss='mse', optimizer=self.optimizer)
        self.model.compile(loss=huberloss, optimizer=self.optimizer)
 
    # 重みの学習
    def replay(self, memory, batch_size, gamma, targetQN):
        #inputs = np.zeros((batch_size, 4))
        inputs = np.zeros((batch_size, 6))
        #targets = np.zeros((batch_size, 2))
        targets = np.zeros((batch_size, RSPEED_BIN*RSPEED_BIN))
        mini_batch = memory.sample(batch_size)
 
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            #print(inputs)
            target = reward_b
 
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                #print(next_state_b)
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]
                #print(target)
                
            targets[i] = self.model.predict(state_b)    # Qネットワークの出力
            #print(action_b)
            targets[i][action_b] = target               # 教師信号
            #print(targets)
            self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定
 
 
# [3]Experience ReplayとFixed Target Q-Networkを実現するメモリクラス
class Memory:
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
 
    def add(self, experience):
        self.buffer.append(experience)
 
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]
 
    def len(self):
        return len(self.buffer)
 
 
# [4]カートの状態に応じて、行動を決定するクラス
class Actor:
    def get_action(self, state, episode, targetQN):   # [C]ｔ＋１での行動を返す
        # 徐々に最適行動のみをとる、ε-greedy法
        #epsilon = 0.001 + 0.9 / (1.0+episode)
        epsilon = 0.001 + 90 / (1.0+episode)
 
        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = targetQN.model.predict(state)[0]
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
 
        else:
        #    action = np.random.choice([0,1,2,3,4,5,6,7,8,9,
	#		10,11,12,13,14,15,16,17,18,19,
	#		20,21,22,23,24,25,26,27,28,29,
	#		30,31,32,33,34,35,36,37,38,39,
	#		40,41,42,43,44,45,46,47,48,49,
	#		50,51,52,53,54,55,56,57,58,59,
	#		60,61,62,63])  # ランダムに行動する
            action = np.random.choice([0,1,2,3,4,5,6,7,8,9,
			10,11,12,13,14,15])
 
            #print('action-1:%d' % (action))
        return action

##################################################
# 連続値を離散化した(min?maxをnum分割した値)を返す
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
	#print('x:%f, y:%f, angle:%f, 1:%f, 2:%f, 3:%f, 4:%f, 5:%f' % (cart_x, cart_y, cart_angle, sensor1, sensor2, sensor3, sensor4, sensor5))

	# 最大値と最小値
	cart_x_high, cart_y_high, cart_angle_high, \
		sensor1_high, sensor2_high, sensor3_high, sensor4_high, sensor5_high = observation_space.high
	cart_x_low,  cart_y_low,  cart_angle_low,  \
		sensor1_low,  sensor2_low,  sensor3_low, sensor4_low, sensor5_low  = observation_space.low
	#print('high1:%f, high2:%f' % (sensor1_high, sensor2_high))
	#print('low1:%f, low2:%f' % (sensor1_low, sensor2_low))
	#print('sensor1:%f, low1:%f, low2:%f, SENSOR_BIN:%d' % (sensor1, sensor1_low, sensor1_high, SENSOR_BIN))
	
	# ビンの位置(インデックス)を返す
	digitized = [	np.digitize(cart_angle,	bins(cart_angle_low, cart_angle_high, CART_ANGLE_BIN)),
					np.digitize(sensor1,	bins(sensor1_low,    sensor1_high,    SENSOR_BIN)    ),
					np.digitize(sensor2,	bins(sensor2_low,    sensor2_high,    SENSOR_BIN)    ),
					np.digitize(sensor3,	bins(sensor3_low,    sensor3_high,    SENSOR_BIN)    ),
					np.digitize(sensor4,	bins(sensor4_low,    sensor4_high,    SENSOR_BIN)    ),
					np.digitize(sensor5,	bins(sensor5_low,    sensor5_high,    SENSOR_BIN)    )
				]
	
	#return digitized
	return np.array([np.digitize(cart_angle, bins(cart_angle_low, cart_angle_high, CART_ANGLE_BIN)),
			np.digitize(sensor1,    bins(sensor1_low,    sensor1_high,    SENSOR_BIN)    ),
			np.digitize(sensor2,    bins(sensor2_low,    sensor2_high,    SENSOR_BIN)    ),
			np.digitize(sensor3,    bins(sensor3_low,    sensor3_high,    SENSOR_BIN)    ),
			np.digitize(sensor4,    bins(sensor4_low,    sensor4_high,    SENSOR_BIN)    ),
			np.digitize(sensor5,    bins(sensor5_low,    sensor5_high,    SENSOR_BIN)    )
			]) 

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
#def get_action_argmax(q_table, state):
#
#	new_actions = q_table[state]
#	#print(new_actions)
#
#	qmax = -1e300
#	max_i, max_j = -1, -1
#	for i in range(new_actions.shape[0]):
#		for j in range(new_actions.shape[1]):
#			if new_actions[i, j] > qmax:
#				qmax = new_actions[i, j] 
#				max_i = i
#				max_j = j
#
#	return np.array([max_i, max_j])
#	
##################################################
# アクションを離散値から連続値に変換する
##################################################
def d2a_action_argmax(action, action_space):

	#def_left = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,
	#		4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7]
	#def_right = [0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,
	#		0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]

	def_left = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
	def_right = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]

	# 左ホイールの速度, 右ホイールの速度
	#left_rspeed_bin, right_rspeed_bin = action
	left_rspeed_bin = def_left[action]
	right_rspeed_bin = def_right[action]
	
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
#def get_action(	q_table, state, 
#				action, action_space, 
#				observation, observation_space, 
#				reward, episode):
#
#	#new_state = digitize_state(observation, observation_space)
#	#next_state = new_state + state
#	#del next_state[STATE_NUM*(STATE_TIME_NUM-1) : STATE_NUM*STATE_TIME_NUM]
#	next_state = digitize_state(observation, observation_space)
#	action_bin = digitize_action(action, action_space)
#	
#	# ε-greedy(epsilon-greedy)法で次のアクション取得
#	epsilon = 0.5 * (0.999 ** episode)
#	if epsilon <= np.random.uniform(0, 1):
#		# 次の状態から最善のアクションを取得
#		next_action = get_action_argmax(q_table, next_state)
#	else:
#		next_action = np.random.randint(0, RSPEED_BIN, 2)
#	
#	i1 = state + [action_bin[0], action_bin[1]]
#	i2 = next_state + [next_action[0], next_action[1]]
#	q_table[i1] = (1.0 - ALPHA) * q_table[i1] + \
#					ALPHA * (reward + GAMMA * q_table[i2])
#
#	return next_action, next_state
	
##################################################
# メイン
##################################################
if __name__ == '__main__':

	#gamma = 0.99
	gamma = 0.0000099
	islearned = 0	
	# 環境読み込み
	env = Yusuke1goEnv(MAX_STEP)
	observation_space = env.observation_space
	action_space = env.action_space

	# [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
	mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)     # メインのQネットワーク
	targetQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)   # 価値を計算するQネットワーク
	# plot_model(mainQN.model, to_file='Qnetwork.png', show_shapes=True)        # Qネットワークの可視化
	memory = Memory(max_size=memory_size)
	actor = Actor()


	# 強化学習開始
	for episode in range(NUM_EPISODES):
		
		# 環境初期化
		observation = env.reset()
		#env.reset()

		observation, reward, done, _ = env.step(env.action_space.sample())  # 1step目は適当な行動をとる
		state_bin = digitize_state(observation, observation_space)
		#state = np.reshape(state, [1, 8])   # list型のstateを、1行8列の行列に変換
		state_bin = np.reshape(state_bin, [1, 6])   # list型のstateを、1行8列の行列に変換
		#print('state1:%d, state2:%d, state3:%d, state4:%d, state5:%d, state6:%d' % (state_bin[0,0], state_bin[0,1], state_bin[0,2], state_bin[0,3], state_bin[0,4], state_bin[0,5]))
		#print(state_bin)

 
		targetQN = mainQN   # 行動決定と価値計算のQネットワークをおなじにする	

		# 描画有無決定
		isRendering = False
		#if (episode > 20000) and (episode % 200) == 0:
		#	isRendering = True
		#elif (episode > 5000) and (episode % 500) == 0:
		#	isRendering = True
		#elif (episode > 500) and (episode % 1000) == 0:
		#	isRendering = True
		if (episode > 60) and (episode % 50) == 0:
			isRendering = True



		# 最大ステップ数まで学習
		for step in range(MAX_STEP):
			
			# 描画
			if isRendering:
				env.render()
				#wrappers.Monitor
			
			#action_bin = actor.get_action(state, episode, mainQN)   # 時刻tでの行動を決定する
			action_bin = actor.get_action(state_bin, episode, mainQN)   # 時刻tでの行動を決定する
			#print('action_bin:%d' % (action_bin))
			action = d2a_action_argmax(action_bin, action_space)
			#next_state, reward, done, info = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
			observation, reward, done, info = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
			#print('reward:%d' % (reward))

			next_state_bin = digitize_state(observation, observation_space)
			next_state_bin = np.reshape(next_state_bin, [1, 6])     # list型のstateを、1行8列の行列に変換
	
 
			#memory.add((state, action, reward, next_state))     # メモリの更新する
			memory.add((state_bin, action_bin, reward, next_state_bin))     # メモリの更新する
			state_bin = next_state_bin  # 状態更新

 
			# Qネットワークの重みを学習・更新する replay
			if (memory.len() > batch_size) and not islearned:
				mainQN.replay(memory, batch_size, gamma, targetQN)
 
			if DQN_MODE:
				targetQN = mainQN  # 行動決定と価値計算のQネットワークをおなじにする
 
	
			# 終了の場合
			if done:
				#ave_step = ave_step + (step + 1)
				#steps_history.append(step+1)
				#print('Episode-%08d finished at %03d steps' % (episode, step))
				print('%08d,%03d,%d' % (episode, step, observation[1]))
				#if (episode % 500) == 0:
					#ave_step = ave_step / 500
					#ave_reward = ave_reward / 500
				    #print('Episode-%08d finished at %03d steps' % (episode, step))
					#print('Episode-%08d finished at average %03d steps, average rewards %7.1f' 
							#% (episode, ave_step, ave_reward))
					#ave_step = 0
					#ave_reward = 0

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

