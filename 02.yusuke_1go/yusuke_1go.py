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

    def __init__(self):

        # 境界
        self.xmax = 100
        self.xmin = -100
        self.ymax = 1000
        self.ymin = 0

        # カートとホイールのサイズ
        self.cart_width = 20
        self.cart_length = 30
        self.wheel_radius = 8
        self.wheel_width = 6
        self.light_width = 4
        self.light_height = 2

        # 描画サイズ
        self.screen_width = 400
        self.screen_height = 800
        self.world_width = (self.xmax - self.xmin) * 1.1
        self.scale = self.screen_width / self.world_width
        

        # 乱数・描画・状態・ステップ初期化
        self.seed()
        self.viewer = None
        self.state = None
        self.steps = None
        self.max_step = 500

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        # 状態と行動を取得
        x, y, angle = self.state
        left_rspeed, right_rspeed = action

        # 状態更新
        angle_dot = (left_rspeed - right_rspeed) / float(self.cart_width)
        new_angle = angle + angle_dot
        distance = 0.5 * (right_rspeed + left_rspeed)
        new_x = x + distance * math.sin(angle + angle_dot/2)
        new_y = y + distance * math.cos(angle + angle_dot/2)
        self.state = (new_x, new_y, new_angle)

        # y座標の増分に応じて報酬を増やす
        reward = new_y - y

        done = False

        # 終了判定(境界オーバー)
        if new_x > self.xmax:
            done = True
            reward = -500
        elif new_x < self.xmin:
            done = True
            reward = -500
        if new_y < self.ymin:
            done = True
            reward = -500

        # 終了判定(ゴール)
        if new_y > self.ymax:
            done = True
            reward = 1000

        # 終了判定(ステップ数)
        self.steps = self.steps + 1
        if self.steps >= self.max_step :
            done = True
        
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.steps = 0
        #x, y = self.np_random.uniform(low=-50, high=0.50, size=(2,))
        x = self.np_random.uniform(low=-50, high=50)
        y = 0
        angle =  self.np_random.uniform(low=-0.5, high=0.5)
        self.state = np.array([x, y, angle])
        #self.state =  self.np_random.uniform(low=-0.5, high=0.5, size=(3,))
        return self.state
        #return np.array(self.state)

    def render(self, mode='human'):

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

            # ライトの設定
            light_x = self.screen_scale(self.light_width)
            light_y = self.screen_scale(self.light_height)
            left, right, top, bottom = -light_x, light_x, -light_y, light_y
            light = rendering.FilledPolygon([(left,bottom), (left,top), (right,top), (right,bottom)])
            light.set_color(.5,.5,.5)
            light_offset_y = cart_y
            light_trans = rendering.Transform(translation=(0, light_offset_y))
            light.add_attr(light_trans)
            light.add_attr(self.cart_trans)
            self.viewer.add_geom(light)

            # スタートラインの設定
            line_length = self.screen_scale(self.xmax - self.xmin)
            start_line = rendering.Line((0, 0), (line_length, 0))
            start_line.set_color(0,0,0)
            self.start_line_trans = rendering.Transform()
            start_line.add_attr(self.start_line_trans)
            self.viewer.add_geom(start_line)

            # ゴールラインの設定
            line_length = self.screen_scale(self.xmax - self.xmin)
            goal_line = rendering.Line((0, 0), (line_length, 0))
            goal_line.set_color(0,1,0)
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

        if self.state is None: return None

        # カートの描画
        x, y, direction = self.state
        cart_x, cart_y = self.screen_xy(x, y, 0, y)
        self.cart_trans.set_translation(cart_x, cart_y)
        self.cart_trans.set_rotation(-direction)

        # スタートラインの描画
        start_line_x, start_line_y = self.screen_xy(self.xmin, 0, 0, y)
        self.start_line_trans.set_translation(start_line_x, start_line_y)

        # 通過ラインの描画
        list_len = len(self.passsing_line_trans_list)
        for i in range(list_len):
            line_y = self.ymax * float(i+1) / float(list_len+1)
            passing_line_x, passing_line_y = self.screen_xy(self.xmin, line_y, 0, y)
            self.passsing_line_trans_list[i].set_translation(passing_line_x, passing_line_y)
        
        # ゴールラインの描画
        goal_line_x, goal_line_y = self.screen_xy(self.xmin, self.ymax, 0, y)
        self.goal_line_trans.set_translation(goal_line_x, goal_line_y)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

    def screen_scale(self, phy_x):
        x = phy_x * self.scale
        return(x)

    # 物理座標からスクリーン座標に変換する
    def screen_xy(self, phy_x, phy_y, center_x, center_y):
        # 物理座標(center_x, center_y)を画面中央にする
        screen_x = (phy_x - center_x) * self.scale + self.screen_width/2
        screen_y = (phy_y - center_y) * self.scale + self.screen_height/2
        return (screen_x, screen_y)

