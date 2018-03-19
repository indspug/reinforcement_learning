# -*- coding: utf-8 -*-
#
"""
OpenAI Gymの自作環境「ユースケ1号」
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.classic_control import rendering
import numpy as np

class Yusuke1goEnv(gym.Env):

    ##################################################
    # コンストラクタ
    ##################################################
    def __init__(self, max_step=500):

        # 最大ステップ数設定
        self.max_step = max_step

        # 行動と状態の最小・最大値
        self.xmax = 100
        self.xmin = -100
        self.ymax = 1000
        self.ymin = 0
        self.dir_max =  math.pi
        self.dir_min = -math.pi
        self.rspeed_max = 10.0
        self.rspeed_min = 1.0

        # カートとホイールのサイズ
        self.cart_width = 20
        self.cart_length = 30
        self.wheel_radius = 8
        self.wheel_width = 6
        self.head_width = 4
        self.head_height = 2

        # センサーの設定
        self.sensor_range = 150
        self.sensor_val_max = 100
        self.sensor_val_min = 0
        self.sensor_num = 3
        self.sensor_angle = np.array([-math.pi/4, 0, math.pi/4])
        #self.sensor_num = 5
        #self.sensor_angle = np.array([-math.pi/2, -math.pi/4, 0, math.pi/4, math.pi/2])

        # 障害物の設定
        self.obstacle_width = 15
        self.obstacle_height = 15
        self.obstacle_xmax = self.xmax - self.obstacle_width
        self.obstacle_xmin = self.xmin + self.obstacle_width
        self.obstacle_ymax = self.ymax - 100
        self.obstacle_ymin = self.ymin + 100
        self.obstacle_num = 5

        # 描画サイズ
        self.screen_width = 400
        self.screen_height = 800
        self.world_width = (self.xmax - self.xmin) * 1.1
        self.scale = self.screen_width / self.world_width
        
        # 行動と状態の範囲
        action_high = np.array([self.rspeed_max, self.rspeed_max])
        action_low  = np.array([self.rspeed_min, self.rspeed_min])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        state_high = np.array([ self.xmax, self.ymax, self.dir_max, 
                                self.sensor_val_max, self.sensor_val_max, self.sensor_val_max])
                                #self.sensor_val_max, self.sensor_val_max, self.sensor_val_max,
                                #self.sensor_val_max, self.sensor_val_max])
        state_low  = np.array([ self.xmin, self.ymin, self.dir_min, 
                                self.sensor_val_min, self.sensor_val_min, self.sensor_val_min])
                                #self.sensor_val_min, self.sensor_val_min, self.sensor_val_min,
                                #self.sensor_val_min, self.sensor_val_min])
        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)

        # 乱数・描画・状態・ステップ初期化
        self.seed()
        self.viewer = None
        self.obstacle_trans_list = None
        self.state = None
        self.steps = None

    ##################################################
    # 乱数種初期化
    ##################################################
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    ##################################################
    # 1ステップ実行
    ##################################################
    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        # 終了判定初期化        
        done = False

        # 状態と行動を取得
        #x, y, angle, sensor1, sensor2, sensor3 = self.state
        x, y, angle,  = self.state[0], self.state[1], self.state[2]
        left_rspeed, right_rspeed = action

        # 状態更新
        angle_dot = (left_rspeed - right_rspeed) / float(self.cart_width)
        new_angle = self.correct_angle(angle + angle_dot)
        distance = 0.5 * (right_rspeed + left_rspeed)
        new_x = x + distance * math.sin(angle + angle_dot/2)
        new_y = y + distance * math.cos(angle + angle_dot/2)
        
        # 障害物との距離と角度から、センサー値を計算する
        # 障害物にカートが衝突した場合は終了する
        sensor_val = np.full(self.sensor_num, self.sensor_val_min)
        for i in range(len(self.obstacle_xy_list)):
            ox, oy = self.obstacle_xy_list[i]   # 障害物の中心位置
            diff_x = ox - new_x                 # カートと障害物のX位置の差
            diff_y = oy - new_y                 # カートと障害物のy位置の差

            # カートと障害物の距離
            distance = math.sqrt( pow(diff_x,2) + pow(diff_y,2) ) 
            
            # カートと障害物の距離が一定値以下の場合は衝突したとみなす
            if distance < (self.cart_length + self.obstacle_height)/2 :
                done = True
                reward = -1000
                break

            # カートと障害物の距離がセンサーの射程内のとき
            if distance < self.sensor_range:
                # カートと障害物の相対角度
                diff_angle = self.get_diff_angle(new_x, new_y, ox, oy) 

                # カートの向きも考慮したカートと障害物の相対角度
                diff_angle = self.correct_angle(diff_angle - new_angle)

                for j in range(self.sensor_num):
                    #print('(%d,%d) : cart(%.1f,%.1f), obst(%.1f,%.1f)' % 
                    #        (i, j, new_x, new_y, ox, oy))
                    #print('(%d,%d) : cart_angle=%.1f, diff_angle=%.1f' % 
                    #        (i, j, new_angle*180/math.pi, diff_angle*180/math.pi))

                    # センサが障害物の方を向いていれば距離に応じたセンサー値とする
                    # (距離が近いほど大きく、遠いほど小さく)
                    diff_sensor_angle = self.correct_angle(diff_angle - self.sensor_angle[j])
                    if abs(diff_sensor_angle) < math.pi/9:
                        sensor_val_tmp = self.sensor_val_max * (1 - distance/self.sensor_range)
                        if sensor_val[j] < sensor_val_tmp:
                            sensor_val[j] = sensor_val_tmp

        # 状態更新
        self.state = (new_x, new_y, new_angle) + tuple(sensor_val.tolist())

        # 障害物に衝突していない場合は終了判定と報酬の設定を行う
        if not done:

            # y座標の増分に応じて報酬を増やす
            reward = new_y - y
            
            # 終了判定(境界オーバー)
            if new_x > self.xmax:
                done = True
                reward = -1000
            elif new_x < self.xmin:
                done = True
                reward = -1000
            if new_y < self.ymin:
                done = True
                reward = -1000

            # 終了判定(ゴール)
            if new_y > self.ymax:
                done = True
                reward = 1000

            # 終了判定(ステップ数)
            self.steps = self.steps + 1
            if self.steps >= self.max_step :
                done = True
        
        return np.array(self.state), reward, done, {}

    ##################################################
    # リセット
    ##################################################
    def reset(self):
        self.steps = 0
        x = self.np_random.uniform(low=self.xmin/2, high=self.xmax/2)
        y = self.ymin
        angle = self.np_random.uniform(low=-math.pi/4, high=math.pi/4)
        sensor_val = np.full(self.sensor_num, self.sensor_val_min)
        self.state = (x, y, angle) + tuple(sensor_val.tolist())

        # 描画環境の初期化
        if self.viewer is None:
            self.init_viewer()

        # 障害物の位置をリセット
        self.reset_obstacle()

        return np.array(self.state)

    ##################################################
    # 描画
    ##################################################
    def render(self, mode='human'):

        # 状態が未初期化のときは終了
        if self.state is None: return None

        # カートの描画
        #cart_x, cart_y, direction, sensor1, sensor2, sensor3 = self.state
        cart_x, cart_y, direction = self.state[0], self.state[1], self.state[2]
        screen_x, screen_y = self.screen_xy(cart_x, cart_y, 0, cart_y)
        self.cart_trans.set_translation(screen_x, screen_y)
        self.cart_trans.set_rotation(-direction)

        # スタートラインの描画
        start_line_x, start_line_y = self.screen_xy(self.xmin, 0, 0, cart_y)
        self.start_line_trans.set_translation(start_line_x, start_line_y)

        # 通過ラインの描画
        list_len = len(self.passsing_line_trans_list)
        for i in range(list_len):
            line_y = self.ymax * float(i+1) / float(list_len+1)
            passing_line_x, passing_line_y = self.screen_xy(self.xmin, line_y, 0, cart_y)
            self.passsing_line_trans_list[i].set_translation(passing_line_x, passing_line_y)
        
        # ゴールラインの描画
        goal_line_x, goal_line_y = self.screen_xy(self.xmin, self.ymax, 0, cart_y)
        self.goal_line_trans.set_translation(goal_line_x, goal_line_y)

        # 障害物の描画
        for i, obstacle_trans in enumerate(self.obstacle_trans_list):
        #for i in range(len(self.obstacle_trans_list)):
            obstacle_trans = self.obstacle_trans_list[i]
            x, y = self.obstacle_xy_list[i]
            screen_x, screen_y = self.screen_xy(x, y, 0, cart_y)
            obstacle_trans.set_translation(screen_x, screen_y)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    ##################################################
    # 環境終了
    ##################################################
    def close(self):
        if self.viewer: self.viewer.close()

    ##################################################
    # 描画環境を初期化する
    ##################################################
    def init_viewer(self):

        if self.viewer is None:

            # Viewer生成
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            # カートの設定
            cart_x = self.screen_scale(self.cart_width/2)
            cart_y = self.screen_scale(self.cart_length/2)
            left, right, top, bottom = -cart_x, cart_x, -cart_y, cart_y
            cart = rendering.FilledPolygon([(left,bottom), (left,top), (right,top), (right,bottom)])
            cart.set_color(.5,.5,.5)
            self.cart_trans = rendering.Transform()
            cart.add_attr(self.cart_trans)
            self.viewer.add_geom(cart)

            # 左ホイールの設定
            wheel_x = self.screen_scale(self.wheel_width/2)
            wheel_y = self.screen_scale(self.wheel_radius)
            left, right, top, bottom = -wheel_x, wheel_x, -wheel_y, wheel_y
            left_wheel = rendering.FilledPolygon([(left,bottom), (left,top), (right,top), (right,bottom)])
            left_wheel.set_color(.5,.5,.5)
            left_wheel_offset_x = -cart_x
            left_wheel_trans = rendering.Transform(translation=(left_wheel_offset_x, 0))
            left_wheel.add_attr(left_wheel_trans)
            left_wheel.add_attr(self.cart_trans)
            self.viewer.add_geom(left_wheel)

            # 右ホイールの設定
            right_wheel = rendering.FilledPolygon([(left,bottom), (left,top), (right,top), (right,bottom)])
            right_wheel.set_color(.5,.5,.5)
            right_wheel_offset_x = cart_x
            right_wheel_trans = rendering.Transform(translation=(right_wheel_offset_x, 0))
            right_wheel.add_attr(right_wheel_trans)
            right_wheel.add_attr(self.cart_trans)
            self.viewer.add_geom(right_wheel)

            # 先端の設定
            head_x = self.screen_scale(self.head_width)
            head_y = self.screen_scale(self.head_height)
            left, right, top, bottom = -head_x, head_x, -head_y, head_y
            head = rendering.FilledPolygon([(left,bottom), (left,top), (right,top), (right,bottom)])
            head.set_color(.5,.5,.5)
            head_offset_y = cart_y
            head_trans = rendering.Transform(translation=(0, head_offset_y))
            head.add_attr(head_trans)
            head.add_attr(self.cart_trans)
            self.viewer.add_geom(head)

            # センサビームの設定
            for i in range(self.sensor_num):
                beam_length = self.screen_scale(self.sensor_range)
                beam = rendering.PolyLine([(0, 0), (0, beam_length)], False)
                beam.set_color(1.0,0.8,0)
                beam.set_linewidth(2)
                beam_trans = rendering.Transform()
                beam_trans.set_rotation(self.sensor_angle[i])
                beam.add_attr(beam_trans)
                beam.add_attr(head_trans)
                beam.add_attr(self.cart_trans)
                self.viewer.add_geom(beam)

            # スタートラインの設定
            line_length = self.screen_scale(self.xmax - self.xmin)
            start_line = rendering.PolyLine([(0, 0), (line_length, 0)], False)
            start_line.set_color(0,0,0)
            start_line.set_linewidth(3)
            self.start_line_trans = rendering.Transform()
            start_line.add_attr(self.start_line_trans)
            self.viewer.add_geom(start_line)

            # ゴールラインの設定
            line_length = self.screen_scale(self.xmax - self.xmin)
            goal_line = rendering.PolyLine([(0, 0), (line_length, 0)], False)
            goal_line.set_color(0,1,0)
            goal_line.set_linewidth(3)
            self.goal_line_trans = rendering.Transform()
            goal_line.add_attr(self.goal_line_trans)
            self.viewer.add_geom(goal_line)

            # 通過ラインの設定
            self.passsing_line_trans_list = []
            line_length = self.screen_scale(self.xmax - self.xmin)
            for i in range(4):
                passsing_line = rendering.Line((0, 0), (line_length, 0))
                passsing_line.set_color(0,0,0)
                passsing_line_trans = rendering.Transform()
                passsing_line.add_attr(passsing_line_trans)
                passing_line_style = rendering.LineStyle(0xF0F0)
                passsing_line.add_attr(passing_line_style)
                self.viewer.add_geom(passsing_line)
                self.passsing_line_trans_list.append(passsing_line_trans)

            # 左境界線の設定
            boundary_x, boundary_y = self.screen_xy(self.xmin, 0, 0, 0)
            left_boundary = rendering.Line((boundary_x, 0), (boundary_x, self.screen_height))
            left_boundary.set_color(1,0,0)
            self.viewer.add_geom(left_boundary)

            # 右境界線の設定
            boundary_x, boundary_y = self.screen_xy(self.xmax, 0, 0, 0)
            right_boundary = rendering.Line((boundary_x, 0), (boundary_x, self.screen_height))
            right_boundary.set_color(1,0,0)
            self.viewer.add_geom(right_boundary)

    ##################################################
    # 障害物の位置をリセットする
    ##################################################
    def reset_obstacle(self):

        # 障害物の描画の初期設定
        if self.obstacle_trans_list is None:
            self.obstacle_trans_list = []
            for i in range(self.obstacle_num):
                w = self.screen_scale(self.obstacle_width/2)
                h = self.screen_scale(self.obstacle_height/2)
                left, right, top, bottom = -w, w, -h, h
                obstacle = rendering.FilledPolygon([(left,bottom), (left,top), (right,top), (right,bottom)])
                obstacle.set_color(1,0,0)
                obstacle_trans = rendering.Transform()
                obstacle.add_attr(obstacle_trans)
                self.viewer.add_geom(obstacle)
                self.obstacle_trans_list.append(obstacle_trans)

        # 障害物の位置をリセット
        self.obstacle_xy_list = []
        for i, obstacle_trans in enumerate(self.obstacle_trans_list):
            x = self.np_random.uniform(low=self.obstacle_xmin, high=self.obstacle_xmax)
            y = self.np_random.uniform(low=self.obstacle_ymin, high=self.obstacle_ymax)
            self.obstacle_xy_list.append((x, y))

    ##################################################
    # 角度(radian)を-180°〜180°に補正する
    ##################################################
    def correct_angle(self, rad):
        angle = rad
        while angle > math.pi:
            angle = angle - math.pi
        while angle < -math.pi:
            angle = angle + math.pi

        return angle

    ##################################################
    # 2点間の角度差を取得する
    ##################################################
    def get_diff_angle(self, self_x, self_y, opponent_x, opponent_y):
        diff_x = opponent_x - self_x
        diff_y = opponent_y - self_y

        diff_angle = math.atan2(abs(diff_x), abs(diff_y))
        if (diff_x > 0) and (diff_y < 0):   # 第二象限
            diff_angle = math.pi - diff_angle
        elif (diff_x < 0) and (diff_y < 0): # 第三象限
            diff_angle = diff_angle - math.pi
        elif (diff_x < 0) and (diff_y > 0): # 第四象限
             diff_angle = -diff_angle

        return diff_angle

    ##################################################
    # 物理座標からスクリーン座標に変換する
    ##################################################
    def screen_scale(self, phy_x):
        x = phy_x * self.scale
        return(x)

    ##################################################
    # 物理座標からスクリーン座標に変換する
    # 指定した(center_x, center_y)画面中央に来るように変換する
    ##################################################
    def screen_xy(self, phy_x, phy_y, center_x, center_y):
        # 物理座標(center_x, center_y)を画面中央にする
        screen_x = (phy_x - center_x) * self.scale + self.screen_width/2
        screen_y = (phy_y - center_y) * self.scale + self.screen_height/2
        return (screen_x, screen_y)

