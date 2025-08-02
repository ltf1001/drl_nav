#! /usr/bin/python2.7
#coding:utf-8
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
# Licensed under the Apache License, Version 2.0 (the "License");
#################################################################################

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn
import torch
import torch.nn as nn
import torch.nn.functional as F


# 扰动网络：对原始激光雷达数据进行扰动（模拟遮挡或噪声）
class PerturbationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        """
        input_dim: 激光雷达点数量（如360个点）
        hidden_dim: 隐藏层维度
        """
        super(PerturbationNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc_mask = nn.Linear(hidden_dim, input_dim)  # 输出遮挡掩码概率
        self.fc_noise = nn.Linear(hidden_dim, input_dim)  # 输出噪声幅度

    def forward(self, x):
        """
        x: 原始激光雷达数据（tensor，形状为[1, input_dim]）
        返回: 
            mask_prob: 每个点被遮挡的概率（0-1）
            noise: 每个点的噪声值（用于模拟测量误差）
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mask_prob = torch.sigmoid(self.fc_mask(x))  # 遮挡概率（0-1）
        noise = torch.tanh(self.fc_noise(x)) * 0.2  # 噪声范围[-0.2, 0.2]
        return mask_prob, noise


class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.obstacle_min_range = 0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        
        # 初始化扰动网络（动态获取激光雷达点数）
        self.perturb_net = None  # 延迟初始化（等待获取激光雷达数据后确定输入维度）
        self.perturb_prob = 0.3  # 30%概率使用扰动数据
        self.lidar_input_dim = None  # 激光雷达点数量

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)
        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi
        elif heading < -pi:
            heading += 2 * pi
        self.heading = round(heading, 2)

    def getState(self, scan, num_sectors=24):
        """处理激光雷达数据，加入扰动后构建状态向量"""
        # 初始化扰动网络（首次调用时确定激光雷达点数）
        if self.lidar_input_dim is None:
            self.lidar_input_dim = len(scan.ranges)
            self.perturb_net = PerturbationNetwork(input_dim=self.lidar_input_dim)
            rospy.loginfo("初始化扰动网络，激光雷达点数: %d" % self.lidar_input_dim)

        # 获取原始激光雷达数据
        original_ranges = np.array(scan.ranges)
        ranges = original_ranges.copy()

        # 随机触发扰动（30%概率）
        if np.random.rand() < self.perturb_prob and self.perturb_net is not None:
            # 将原始数据转换为tensor
            x = torch.tensor(original_ranges, dtype=torch.float32).unsqueeze(0)  # 形状[1, input_dim]
            
            # 生成扰动参数
            mask_prob, noise = self.perturb_net(x)
            mask = (mask_prob > 0.6).squeeze().numpy()  # 遮挡掩码（60%概率阈值）
            noise_np = noise.squeeze().detach().numpy()  # 噪声值

            # 1. 遮挡（将部分点设为无穷远，模拟视线受阻）
            ranges[mask] = float('inf')
            # 2. 加噪声（模拟测量误差）
            non_mask = ~mask
            ranges[non_mask] = np.clip(ranges[non_mask] + noise_np[non_mask], 0.05, 3.5)  # 限制范围

        # 后续处理与原始逻辑一致（将激光雷达数据划分为扇区）
        sector_min_ranges = [float('inf')] * num_sectors
        heading = self.heading
        min_range = 0.15  # 碰撞阈值
        done = False

        for i in range(len(ranges)):
            # 计算每个点对应的角度和扇区
            angle = (i * 2 * math.pi / len(ranges)) - math.pi
            sector_idx = int((angle + math.pi) / (2 * math.pi) * num_sectors)
            sector_idx = min(sector_idx, num_sectors - 1)  # 防止越界

            # 处理扫描值
            if ranges[i] == float('Inf'):
                range_val = 3.5  # 最大检测范围
            elif np.isnan(ranges[i]):
                range_val = 0
            else:
                range_val = ranges[i]

            # 更新扇区最小距离
            if range_val < sector_min_ranges[sector_idx]:
                sector_min_ranges[sector_idx] = range_val

        # 填充未检测到数据的扇区
        for i in range(len(sector_min_ranges)):
            if sector_min_ranges[i] == float('inf'):
                sector_min_ranges[i] = 3.5

        # 碰撞检测
        obstacle_min_range = min(sector_min_ranges)
        if obstacle_min_range < min_range:
            done = True

        # 目标到达检测
        current_distance = math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y)
        if current_distance < 0.2:
            self.get_goalbox = True

        # 构建状态向量
        state = sector_min_ranges + [heading, current_distance, obstacle_min_range, np.argmin(sector_min_ranges)]
        return state, done

    def setReward(self, state, done, action):
        yaw_reward = []
        obstacle_min_range = state[-2]
        self.obstacle_min_range = obstacle_min_range
        current_distance = state[-3]
        heading = state[-4]

        # 角度奖励计算
        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        # 障碍物奖励（避免过近）
        if obstacle_min_range <= 0.2:
            scan_reward = -1/(obstacle_min_range + 0.3)  # 近距离惩罚
        else:
            scan_reward = 2  # 安全距离奖励

        # 距离奖励（靠近目标）
        distance_rate = 2 **(current_distance / self.goal_distance)
        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate) + scan_reward

        # 碰撞惩罚
        if done:
            rospy.loginfo("Collision!!")
            reward = -500 + scan_reward
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.pub_cmd_vel.publish(Twist())

        # 到达目标奖励
        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 1000 + scan_reward
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward

    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        # 发布控制指令
        vel_cmd = Twist()
        vel_cmd.angular.z = ang_vel
        vel_cmd.linear.x = 0.2
        self.pub_cmd_vel.publish(vel_cmd)

        # 获取激光雷达数据
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        # 处理状态和奖励
        state, done = self.getState(data)
        reward = self.setReward(state, done, action)
        return np.asarray(state), reward, done

    def reset(self):
        # 重置仿真环境
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        # 获取初始激光雷达数据
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        # 初始化目标位置
        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)
        return np.asarray(state)