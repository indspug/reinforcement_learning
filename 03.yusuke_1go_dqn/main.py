# -*- coding: utf-8 -*-
#
##################################################
# OpenAI Gymの自作環境「yusuke_1go」をDQNで強化学習する
##################################################

import myenv 
import sys, os
sys.path.append(os.getcwd())
import math
import numpy as np
import gym
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
        inputs = np.zeros((batch_size, 6))
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
            action = np.random.choice([0,1,2,3,4,5,6,7,8,9,
			10,11,12,13,14,15])
 
            #print('action-1:%d' % (action))
        return action

##################################################
#  
##################################################
def min_max(value, min, max):
	return (value - min)/(max - min)

##################################################
# 状態を離散値に変換する
##################################################
def digitize_state(observation, observation_space):

	# カートX位置, カートY位置, カートの角度, センサー値
	cart_x, cart_y, cart_angle, sensor1, sensor2, sensor3, sensor4, sensor5 = observation

	return np.array([min_max(cart_angle, -math.pi, math.pi),
			min_max(sensor1, 0, 100), min_max(sensor2, 0, 100),
			min_max(sensor3, 0, 100), min_max(sensor4, 0, 100),
			min_max(sensor5, 0, 100)])

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
# アクションを離散値から連続値に変換する
##################################################
def d2a_action_argmax(action, action_space):

	def_left = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
	def_right = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]

	# 左ホイールの速度, 右ホイールの速度
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
# メイン
##################################################
if __name__ == '__main__':

	#gamma = 0.99
	#gamma = 0.0000099
	gamma = 0.000099
	islearned = 0	
	# 環境読み込み
	#env = Yusuke1goEnv(MAX_STEP)
	env = gym.make('myenv-v0')
	env = wrappers.Monitor(env, './movie', force=True, video_callable=(lambda ep: ep % 100 == 0))
	observation_space = env.observation_space
	action_space = env.action_space

	# [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
	mainQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)     # メインのQネットワーク
	targetQN = QNetwork(hidden_size=hidden_size, learning_rate=learning_rate)   # 価値を計算するQネットワーク
	memory = Memory(max_size=memory_size)
	actor = Actor()


	# 強化学習開始
	for episode in range(NUM_EPISODES):
		
		# 環境初期化
		observation = env.reset()
		#env.reset()

		observation, reward, done, _ = env.step(env.action_space.sample())  # 1step目は適当な行動をとる
		state_bin = digitize_state(observation, observation_space)
		state_bin = np.reshape(state_bin, [1, 6])   # list型のstateを、1行8列の行列に変換
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
			
			action_bin = actor.get_action(state_bin, episode, mainQN)   # 時刻tでの行動を決定する
			#print('action_bin:%d' % (action_bin))
			action = d2a_action_argmax(action_bin, action_space)
			observation, reward, done, info = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
			#print('reward:%d' % (reward))

			next_state_bin = digitize_state(observation, observation_space)
			next_state_bin = np.reshape(next_state_bin, [1, 6])     # list型のstateを、1行8列の行列に変換
	
 
			memory.add((state_bin, action_bin, reward, next_state_bin))     # メモリの更新する
			state_bin = next_state_bin  # 状態更新

 
			# Qネットワークの重みを学習・更新する replay
			if (memory.len() > batch_size) and not islearned:
				mainQN.replay(memory, batch_size, gamma, targetQN)
 
			if DQN_MODE:
				targetQN = mainQN  # 行動決定と価値計算のQネットワークをおなじにする
 
	
			# 終了の場合
			if done:
				print('%08d,%03d,%d' % (episode, step, observation[1]))

				break
		

	# 環境終了
	env.close()

