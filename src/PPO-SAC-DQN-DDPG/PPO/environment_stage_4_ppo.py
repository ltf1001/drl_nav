#! /usr/bin/python2.7
#coding:utf-8
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

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

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.obstacle_min_range =0.
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
#获取目标点距离
    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance
#获取里程计信息
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
        """
        处理激光雷达扫描数据并构建状态向量
        
        参数:
        scan - 激光雷达扫描数据
        num_sectors - 将360度划分为的扇区数量，默认为24个扇区
        
        返回:
        state - 状态向量
        done - 是否结束(碰撞或到达目标)
        """
        # 初始化扇区最小值数组，每个元素代表一个扇区的最小距离
        sector_min_ranges = [float('inf')] * num_sectors
        heading = self.heading
        min_range = 0.15  # 碰撞距离阈值
        done = False
        
        # 处理每个扫描点
        for i in range(len(scan.ranges)):
            # 角度归一化到[-pi, pi]
            angle = (i * 2 * math.pi / len(scan.ranges)) - math.pi
            # 计算扇区索引
            sector_idx = int((angle + math.pi) / (2 * math.pi) * num_sectors)
            sector_idx = min(sector_idx, num_sectors - 1)  # 防止索引越界
            
            # 处理扫描值
            if scan.ranges[i] == float('Inf'):
                range_val = 3.5  # 最大检测范围
            elif np.isnan(scan.ranges[i]):
                range_val = 0
            else:
                range_val = scan.ranges[i]
            
            # 更新扇区最小距离
            if range_val < sector_min_ranges[sector_idx]:
                sector_min_ranges[sector_idx] = range_val
        
        # 确保所有扇区都有有效值
        for i in range(len(sector_min_ranges)):
            if sector_min_ranges[i] == float('inf'):
                sector_min_ranges[i] = 3.5  # 如果某个扇区没有检测到障碍物
        
        # 检测最近障碍物
        obstacle_min_range = min(sector_min_ranges)
        obstacle_sector = np.argmin(sector_min_ranges)
        
        if obstacle_min_range < min_range:
            done = True
        
        current_distance = math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y)
        
        if current_distance < 0.2:
            self.get_goalbox = True
        
        state = sector_min_ranges + [heading, current_distance, obstacle_min_range, obstacle_sector]
        
        return state, done


    def setReward(self, state, done, action):#传入state,done,action
        yaw_reward = []#角度奖励
        obstacle_min_range = state[-2]#获取激光雷达信息最小的数据
        self.obstacle_min_range = obstacle_min_range#
        current_distance = state[-3]#获取当前数据
        heading = state[-4]#小车的朝向角


        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2#角度分解
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])#角度计算
            yaw_reward.append(tr)#储存角度奖励

        if obstacle_min_range <= 0.2:#激光雷达最小数据小于0.1
            scan_reward = -1/(obstacle_min_range+0.3)#奖励范围-3.33到-2.5
        else:
            scan_reward =2
        # reward = scan_reward
        # return scan_reward
        distance_rate = 2 ** (current_distance / self.goal_distance)#距离比

        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate) +scan_reward
        # reward =scan_reward 

#碰撞
        if done:
            rospy.loginfo("Collision!!")
            reward = -500+scan_reward
            self.goal_x,self.goal_y = self.respawn_goal.getPosition(True,delete=True)
            self.pub_cmd_vel.publish(Twist())
#到达目标点
        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 1000+scan_reward
            self.pub_cmd_vel.publish(Twist())#停止运动
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)#删除模型
            self.goal_distance = self.getGoalDistace()#获得目标点
            self.get_goalbox = False#置False

        return reward


    def step(self, action):
        # obstacle_min_range = state[-2]
        max_angular_vel = 1.5#最大角速度
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5

        # global obstacle_min_range
        vel_cmd = Twist()
        vel_cmd.angular.z = ang_vel
        vel_cmd.linear.x = 0.2


        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward = self.setReward(state, done, action)

        return np.asarray(state), reward, done

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.asarray(state)