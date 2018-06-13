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
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from keras import backend as K
from collections import deque


# パラメータ
MAX_EPISODE = 10000000	# MAXエピソード
MAX_STEP = 1000		# MAXステップ/1エピソード

MOVIE_SAVE_CYCLE = 100	# ムービー保存周期(エピソード単位)
RENDER_CYCLE = 100	# 描画周期(エピソード単位)

STATE_NUM = 6		# 状態数(入力パラメータ数)
			# 角度,センサ1,センサ2,センサ3,センサ4,センサ5
RSPEED_BIN = 4		# 回転速度の離散化数

#ALPHA = 0.02		# 学習率
GAMMA = 0.9		# 割引率

DQN_MODE = 1    # 1がDQN、0がDDQN

HIDDEN_SIZE = [32, 64]	# Q-networkの隠れ層のニューロンの数
LEARNING_RATE = 0.005	# Q-networkの学習係数
memory_size = 10000	# バッファーメモリの大きさ
BATCH_SIZE = 16		# Q-networkを更新するバッチの大きさ
 
RESULT_OUTPUT_CYCLE = 500	# 結果出力サイクル(エピソード単位)
RESULT_CSV = "./result.csv"	# 結果ファイル

WEIGHTS_SAVE_CYCLE = 100	# 重み保存周期(エピソード単位)
WEIGHTS_SAVE_DIR = "./weights"	# 重み保存ディレクトリ


# [1]損失関数の定義
# 損失関数に関数を使用します
def huberloss(y_true, y_pred):
	return K.mean(K.minimum(0.5*K.square(y_pred-y_true), K.abs(y_pred-y_true)-0.5), axis=1)

# [2]Q関数をディープラーニングのネットワークをクラスとして定義
class QNetwork:
	def __init__(self, learning_rate=LEARNING_RATE, 
		state_size=STATE_NUM, action_size=RSPEED_BIN*RSPEED_BIN, 
		hidden_size=[16], ckpt_path=''):
		
		self.state_size = state_size
		self.action_size = action_size
		
		self.model = Sequential()
		self.model.add( InputLayer(input_shape=(state_size,)) )
		for size in hidden_size:
			self.model.add(Dense(size, activation='relu'))
		self.model.add(Dense(action_size, activation='linear'))
	
	# 重みを読み込む
		if ckpt_path:
			self.model.load_weights(ckpt_path)
		
		self.optimizer = Adam(lr=learning_rate)  # 誤差を減らす学習方法はAdam
		self.model.compile(loss='mse', optimizer=self.optimizer)
		#self.model.compile(loss=huberloss, optimizer=self.optimizer)
	
	# 重みの保存
	def save_weights(self, directory, ckpt_name, episode):
		if not os.path.exists(directory):
			os.makedirs(directory)
		ckpt_path = "{0}/{1}_{2:07d}.h5".format(directory, ckpt_name, episode)
		self.model.save_weights(ckpt_path) 
		
	# 重みの学習
	def replay(self, memory, batch_size, gamma, targetQN):
		inputs = np.zeros((batch_size, self.state_size))
		targets = np.zeros((batch_size, self.action_size))
		mini_batch = memory.sample(batch_size)
		
		# 教師信号作成
		for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
			inputs[i:i + 1] = state_b
			target = reward_b
			
			# 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
			retmainQs = self.model.predict(next_state_b)[0]
			next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
			target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]
			    
			targets[i] = self.model.predict(state_b)    # Qネットワークの出力
			targets[i][action_b] = target               # 教師信号
			
		self.model.fit(inputs, targets, epochs=1, verbose=0)
				# epochsは訓練データの反復回数、verbose=0は表示なしの設定
	
	
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
		epsilon = 0.001 + 0.9 / (1.0+episode)
		#epsilon = 0.001 + 90 / (1.0+episode)
		
		if epsilon <= np.random.uniform(0, 1):
			retTargetQs = targetQN.model.predict(state)[0]
			action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
		
		else:
			#action = np.random.choice([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
			action = np.random.randint(0, RSPEED_BIN*RSPEED_BIN)
		
			#print('action-1:%d' % (action))
		return action

##################################################
# 正規化(min-max normalization)
##################################################
def min_max(value, min, max):
	return (value - min)/(max - min)

##################################################
# 状態を正規化する
##################################################
def normalize_state(observation, observation_space):

	# カートX位置, カートY位置, カートの角度, センサー値
	cart_x, cart_y, cart_angle, sensor1, sensor2, sensor3, sensor4, sensor5 = observation

	cart_x_low, cart_y_low, cart_angle_low, \
		sensor1_low, sensor2_low, sensor3_low, \
		sensor4_low, sensor5_low = observation_space.low
        
	cart_x_high, cart_y_high, cart_angle_high, \
		sensor1_high, sensor2_high, sensor3_high, \
		sensor4_high, sensor5_high = observation_space.high

	return np.array([min_max(cart_angle, cart_angle_low, cart_angle_high),
			min_max(sensor1, sensor1_low, sensor1_high), 
			min_max(sensor2, sensor2_low, sensor2_high),
			min_max(sensor3, sensor3_low, sensor3_high), 
			min_max(sensor4, sensor4_low, sensor4_high),
			min_max(sensor5, sensor5_low, sensor5_high)])

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

	# [usage] python3 main.py [ckpt_path] [episode]
	#   
	#   ckpt_path:保存した重みのファイルパスを指定
	#   episode  :学習を再開するエピソードを指定(整数)
	#
	# ex) python3 main.py ./weights/weights_0001000.h5 1001
	
	# コマンドライン引数取得
	argvs = sys.argv
	argc = len(argvs)
	start_episode = 0
	ckpt_path = ""
	if (argc >= 3):
		ckpt_path = argvs[1]		# 保存した重みを取得
		start_episode = int(argvs[2])	# 開始エピソードを取得
	
	# 環境読み込み
	env = gym.make('myenv-v0')
	env = wrappers.Monitor(
		env, './movie', force=True, 
		video_callable=(lambda ep: ep % MOVIE_SAVE_CYCLE == 0))
	observation_space = env.observation_space
	action_space = env.action_space

	# [5.2]Qネットワークとメモリ、Actorの生成--------------------------------------------------------
	mainQN = QNetwork(
		learning_rate=LEARNING_RATE,
		state_size=STATE_NUM, action_size=RSPEED_BIN*RSPEED_BIN, 
		hidden_size=HIDDEN_SIZE,
		ckpt_path=ckpt_path
	)	# メインのQネットワーク
	targetQN = QNetwork(
		learning_rate=LEARNING_RATE,
		state_size=STATE_NUM, action_size=RSPEED_BIN*RSPEED_BIN, 
		hidden_size=HIDDEN_SIZE,
		ckpt_path=ckpt_path
	)	# 価値を計算するQネットワーク
	memory = Memory(max_size=memory_size)
	actor = Actor()
	
	# 結果ファイル生成
	fo = open(RESULT_CSV, "a")
	fo.write("rspeed_bin,%d\n"    % (RSPEED_BIN))
	fo.write("gamma,%f\n"         % (GAMMA))
	fo.write("hidden_size,")
	for size in HIDDEN_SIZE:
		fo.write("%dx"  % (size))
	fo.write("\n")
	fo.write("learning_rate,%f\n" % (LEARNING_RATE))
	fo.write("batch_size,%d\n"    % (BATCH_SIZE))
	fo.write("episode,goal_num,average_y\n")
	fo.close()
	
	# 結果データ初期化
	goal_num = 0;
	average_y = 0.0;
	
	# 強化学習開始
	for episode in range(start_episode, MAX_EPISODE):
		
		# 環境初期化
		observation = env.reset()
		#env.reset()

		observation, reward, done, _ = env.step(env.action_space.sample())  # 1step目は適当な行動をとる
		state = normalize_state(observation, observation_space)
		state = np.reshape(state, [1, STATE_NUM])   # list型のstateを、1行N列の行列に変換

 
		targetQN = mainQN   # 行動決定と価値計算のQネットワークをおなじにする	

		# 描画有無決定
		isRendering = False
		if (episode % RENDER_CYCLE) == 0:
			isRendering = True
		
		# 最大ステップ数まで学習
		for step in range(MAX_STEP):
			
			# 描画
			if isRendering:
				env.render()
			
			action_bin = actor.get_action(state, episode, mainQN)   # 時刻tでの行動を決定する
			#print('action_bin:%d' % (action_bin))
			action = d2a_action_argmax(action_bin, action_space)
			observation, reward, done, info = env.step(action)   # 行動a_tの実行による、s_{t+1}, _R{t}を計算する
			#print('reward:%d' % (reward))

			next_state = normalize_state(observation, observation_space)
			next_state = np.reshape(next_state, [1, STATE_NUM])     # list型のstateを、1行8列の行列に変換
	
 
			memory.add((state, action_bin, reward, next_state))     # メモリの更新する
			state = next_state  # 状態更新
			
			# Qネットワークの重みを学習・更新する replay
			if (memory.len() > BATCH_SIZE):
				mainQN.replay(memory, BATCH_SIZE, GAMMA, targetQN)
 
			#if DQN_MODE:
			#	targetQN = mainQN  # 行動決定と価値計算のQネットワークをおなじにする
 
	
			# 終了の場合
			if done:
				y = observation[1]
				ymax = observation_space.high[1]
				print('%08d,%03d,%d' % (episode, step, y))
				if (y >= ymax):
					goal_num = goal_num + 1
					average_y = average_y + y
				break
		
		# 価値計算ネットワークの更新
		if DQN_MODE:
			targetQN = mainQN  # 行動決定と価値計算のQネットワークをおなじにする
		
		# 重みの保存
		if (episode % WEIGHTS_SAVE_CYCLE) == 0:
			mainQN.save_weights(WEIGHTS_SAVE_DIR, 'weights', episode)
		
		# 結果出力
		if (episode % RESULT_OUTPUT_CYCLE) == 0:
			average_y = average_y / RESULT_OUTPUT_CYCLE
			fo = open(RESULT_CSV, "a")
			fo.write("%d, %d, %f\n" % (episode, goal_num, average_y))
			fo.close()
			goal_num = 0
			average_y = 0	
		
	# 環境終了
	env.close()

