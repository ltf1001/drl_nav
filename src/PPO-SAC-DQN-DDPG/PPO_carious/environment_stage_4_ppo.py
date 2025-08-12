#! /usr/bin/python2.7
#coding:utf-8

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from respawnGoal import Respawn
from goal_model.respawnGoal import Respawn
# from goal_model.fixgoal import Respawn as Respawn

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

    # def getGoalDistace(self):
    #     goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
    #     return goal_distance

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)
        # 确保最小距离不为零
        return max(goal_distance, 0.01)  # 最小距离设为0.01米


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

        sector_min_ranges = [float('inf')] * num_sectors
        heading = self.heading
        min_range = 0.15  # 碰撞距离阈值
        done = False
        
        for i in range(len(scan.ranges)):
            angle = (i * 2 * math.pi / len(scan.ranges)) - math.pi
            sector_idx = int((angle + math.pi) / (2 * math.pi) * num_sectors)
            sector_idx = min(sector_idx, num_sectors - 1)  # 防止索引越界
            if scan.ranges[i] == float('Inf'):
                range_val = 3.5  # 最大检测范围
            elif np.isnan(scan.ranges[i]):
                range_val = 0
            else:
                range_val = scan.ranges[i]
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


    def setReward(self, state, done, action):
        yaw_reward = []  # 朝向奖励
        obstacle_min_range = state[-2]  # 最近障碍物距离
        self.obstacle_min_range = obstacle_min_range
        current_distance = state[-3]  # 当前离目标点的距离
        heading = state[-4]  # 当前朝向与目标朝向的差值
        
        # --- 朝向奖励计算 ---
        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)
        
        # --- 激光避障奖励 ---
        if obstacle_min_range <= 0.2:
            scan_reward = -1.0 / (obstacle_min_range + 0.3)  # 强烈惩罚，范围约 -3.33 到 -2.5
        else:
            scan_reward = 2.0  # 安全情况下的正向奖励

        # --- 角速度惩罚项：防止自转 ---
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1) / 2 - action) * max_angular_vel * 0.5
        angular_penalty = abs(ang_vel) / max_angular_vel  # 范围 0～1
        spin_penalty = -2.0 * angular_penalty  # 最大惩罚 -2.0（可调）


        if self.goal_distance < 0.01:  # 如果目标距离过小
            rospy.logwarn(f"Goal distance too small ({self.goal_distance}), resetting...")
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
        
        # 确保不会除以零
        safe_goal_distance = max(self.goal_distance, 0.01)
        distance_rate = 2 ** (current_distance / safe_goal_distance)

        # --- 距离比率（离目标越远，奖励越小） ---
        distance_rate = 2 ** (current_distance / self.goal_distance)

        # --- 综合奖励计算 ---
        reward = round(yaw_reward[action] * 5, 2) * distance_rate + scan_reward + spin_penalty

        # --- 碰撞检测 ---
        if done:
            rospy.loginfo("Collision!!")
            reward = -500 + scan_reward  # 惩罚为 -500 + scan 惩罚
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.pub_cmd_vel.publish(Twist())

        # --- 达到目标奖励 ---
        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 1000 + scan_reward
            self.pub_cmd_vel.publish(Twist())  # 停止运动
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False

        return reward



    # def setReward(self, state, done, action):#传入state,done,action
    #     yaw_reward = []#角度奖励
    #     obstacle_min_range = state[-2]#获取激光雷达信息最小的数据
    #     self.obstacle_min_range = obstacle_min_range#
    #     current_distance = state[-3]#获取当前数据
    #     heading = state[-4]#小车的朝向角
    #     for i in range(5):
    #         angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2#角度分解
    #         tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])#角度计算
    #         yaw_reward.append(tr)#储存角度奖励
    #     if obstacle_min_range <= 0.2:#激光雷达最小数据小于0.1
    #         scan_reward = -1/(obstacle_min_range+0.3)#奖励范围-3.33到-2.5
    #     else:
    #         scan_reward =2
    #     # reward = scan_reward
    #     # return scan_reward
    #     distance_rate = 2 ** (current_distance / self.goal_distance)#距离比
    #     reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate) +scan_reward
    #     # reward =scan_reward 
    #     if done:
    #         rospy.loginfo("Collision!!")
    #         reward = -500+scan_reward
    #         self.goal_x,self.goal_y = self.respawn_goal.getPosition(True,delete=True)
    #         self.pub_cmd_vel.publish(Twist())
    #     if self.get_goalbox:
    #         rospy.loginfo("Goal!!")
    #         reward = 1000+scan_reward
    #         self.pub_cmd_vel.publish(Twist())#停止运动
    #         self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)#删除模型
    #         self.goal_distance = self.getGoalDistace()#获得目标点
    #         self.get_goalbox = False#置False
    #     return reward

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

    # def reset(self):
    #     rospy.wait_for_service('gazebo/reset_simulation')
    #     try:
    #         self.reset_proxy()
    #     except (rospy.ServiceException) as e:
    #         print("gazebo/reset_simulation service call failed")

    #     data = None
    #     while data is None:
    #         try:
    #             data = rospy.wait_for_message('scan', LaserScan, timeout=5)
    #         except:
    #             pass

    #     if self.initGoal:
    #         self.goal_x, self.goal_y = self.respawn_goal.getPosition()
    #         self.initGoal = False

    #     self.goal_distance = self.getGoalDistace()
    #     state, done = self.getState(data)

    #     return np.asarray(state)

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
        
        # 添加目标点与机器人重合时的重新生成逻辑
        if self.goal_distance < 0.01:  # 如果距离太小
            rospy.logwarn("Initial goal distance too small, regenerating goal...")
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()

        state, done = self.getState(data)

        return np.asarray(state)
