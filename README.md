# drl_nav
1、启动仿真环境：
cd drl
source devel/setup.bash
roslaunch turtlebot3_gazebo turtlebot3_stage_4.launch

2、启动强化学习
cd ~/drl/src/PPO-SAC-DQN-DDPG/PPO
conda activate ppo
python3 PPO.py 

结束 
killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3

查看数据
cd /home/dell/drl/src/PPO-SAC-DQN-DDPG/PPO_carious
tensorboard --logdir runs# drl_nav
