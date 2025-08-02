#!/usr/bin/env python
# coding=UTF-8

from collections import namedtuple
from itertools import count
import os, time
import numpy as np
import rospy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from environment_stage_4_ppo import Env
from std_msgs.msg import Float32MultiArray

# 初始化TensorBoard
tb = SummaryWriter('/home/dell/drl/src/PPO-SAC-DQN-DDPG/PPO_carious/runs')

# 全局参数
gamma = 0.99
render = False
seed = 1
log_interval = 10
num_state = 28
num_action = 5
env = Env(num_action)
torch.manual_seed(seed)

# 定义经验元组
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

# ICM好奇心模块
class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128, feature_dim=128, eta=0.01, beta=0.2, device=None):
        """
        Intrinsic Curiosity Module (ICM)
        
        参数:
        state_dim: 状态维度
        action_dim: 动作空间大小
        hidden_dim: 隐藏层维度
        feature_dim: 特征向量维度
        eta: 内在奖励缩放因子
        beta: 前向损失权重
        device: 计算设备 
        """
        super(ICM, self).__init__()
        # 自动检测设备（优先使用GPU）
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置模型维度参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.eta = eta  # 好奇心奖励缩放因子
        self.beta = beta  # 前向损失权重
        
        # 特征提取网络: state -> feature vector
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.ReLU(),
        ).to(self.device)
        
        # 逆模型: (phi(s_t), phi(s_{t+1})) -> a_t
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(self.device)
        
        # 前向模型: (phi(s_t), a_t) -> predicted phi(s_{t+1})
        self.forward_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        ).to(self.device)
        
        # 损失函数和优化器
        self.MSE = nn.MSELoss()
        self.CE = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def get_intrinsic_reward(self, state, next_state, action):
        """
        计算单个转换的内在好奇心奖励
        """
        # 转换为tensor并添加批次维度
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # 动作处理
        action_tensor = torch.as_tensor([action], dtype=torch.int64, device=self.device)
        action_onehot = F.one_hot(action_tensor, num_classes=self.action_dim).float()
        
        # 提取特征
        state_feature = self.feature_net(state_tensor)
        next_state_feature = self.feature_net(next_state_tensor)
        
        # 预测特征
        forward_input = torch.cat((state_feature, action_onehot), dim=1)
        next_state_feature_pred = self.forward_net(forward_input)
        
        # 内在奖励: 缩放因子 * 1/2 * ||预测特征 - 实际特征||^2
        intrinsic_reward = self.eta * 0.5 * self.MSE(next_state_feature_pred, next_state_feature).item()
        return intrinsic_reward

    def train_icm(self, states, next_states, actions):
        """
        使用一批经验训练ICM模型
        """
        # 将列表转换为numpy数组再转换为张量，避免警告
        states_np = np.array(states)
        next_states_np = np.array(next_states)
        actions_np = np.array(actions)
        
        # 转换数据为tensor
        states_tensor = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.as_tensor(next_states_np, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions_np, dtype=torch.int64, device=self.device)
        
        # 动作转one-hot
        action_onehot = F.one_hot(actions_tensor, num_classes=self.action_dim).float()
        
        # 提取特征
        states_feature = self.feature_net(states_tensor)
        next_states_feature = self.feature_net(next_states_tensor)
        
        # 训练逆模型
        inverse_input = torch.cat((states_feature, next_states_feature), dim=1)
        action_pred_logits = self.inverse_net(inverse_input)
        inverse_loss = self.CE(action_pred_logits, actions_tensor)
        
        # 训练前向模型
        forward_input = torch.cat((states_feature, action_onehot), dim=1)
        next_states_feature_pred = self.forward_net(forward_input)
        forward_loss = self.MSE(next_states_feature_pred, next_states_feature)
        
        # 总损失
        total_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss
        
        # 优化步骤
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item(), inverse_loss.item(), forward_loss.item()

# 策略网络
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob

# 价值网络
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value

# PPO 代理
class PPO(object):
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 128

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.action_loss = 0.
        self.value_loss = 0.
        self.load_models = False
        self.load_ep = 104
        
        # 初始化ICM好奇心模型
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.icm = ICM(
            state_dim=num_state,
            action_dim=num_action,
            eta=0.01,  # 初始内在奖励系数
            beta=0.2,  # 前向损失权重
            device=self.device
        )
        
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        
        # 加载模型
        if self.load_models:
            load_model1 = torch.load("/home/ffd/DRL/PPO/model/maze/98ep.pt")
            self.actor_net.load_state_dict(load_model1['actor_net'])
            self.critic_net.load_state_dict(load_model1['critic_net'])
            print("load model:", str(self.load_ep))
            print("load model successful!!!!!!")
            
        # 将网络移到合适的设备
        self.actor_net = self.actor_net.to(self.device)
        self.critic_net = self.critic_net.to(self.device)

    # 选择动作
    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()

    # 获取状态价值
    def get_value(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    # 保存模型参数
    def save_param(self, e):
        state = {
            'actor_net': self.actor_net.state_dict(),
            'critic_net': self.critic_net.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_net_optimizer.state_dict(),
            'epoch': e,
            'icm': self.icm.state_dict()
        }
        torch.save(state, "/home/dell/drl/src/PPO-SAC-DQN-DDPG/PPO_carious/model/" + str(e) + "ppo.pt")

    # 存储经验
    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    # 更新策略
    def update(self, i_ep):
        # 1. 首先使用当前buffer中的经验训练ICM模型
        if len(self.buffer) > 0:
            # 提取状态、动作和下一个状态
            states = [t.state for t in self.buffer]
            next_states = [t.next_state for t in self.buffer]
            actions = [t.action for t in self.buffer]
            
            # 训练ICM并记录损失
            icm_loss, inverse_loss, forward_loss = self.icm.train_icm(states, next_states, actions)
            tb.add_scalar('ICM/total_loss', icm_loss, i_ep)
            tb.add_scalar('ICM/inverse_loss', inverse_loss, i_ep)
            tb.add_scalar('ICM/forward_loss', forward_loss, i_ep)
        
        # 2. 为每个经验添加好奇心奖励
        modified_buffer = []
        intrinsic_rewards = []
        for t in self.buffer:
            intrinsic_reward = self.icm.get_intrinsic_reward(t.state, t.next_state, t.action)
            intrinsic_rewards.append(intrinsic_reward)
            
            # 原始奖励 = 外部奖励 + 内在奖励
            new_reward = t.reward + intrinsic_reward
            modified_buffer.append(t._replace(reward=new_reward))
        
        # 记录内在奖励统计数据
        if intrinsic_rewards:
            avg_intrinsic_reward = sum(intrinsic_rewards) / len(intrinsic_rewards)
            tb.add_scalar('Reward/Intrinsic_Avg', avg_intrinsic_reward, i_ep)
        
        # 3. 计算回报并更新PPO网络
        # 使用numpy数组避免警告
        states_np = np.array([t.state for t in modified_buffer])
        state = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        
        actions = [t.action for t in modified_buffer]
        action = torch.as_tensor(actions, dtype=torch.int64, device=self.device).view(-1, 1)
        
        rewards = [t.reward for t in modified_buffer]
        old_action_log_prob = torch.as_tensor(
            [t.a_log_prob for t in modified_buffer], 
            dtype=torch.float32, 
            device=self.device
        ).view(-1, 1)

        # 计算折扣回报
        R = 0
        Gt = []
        for r in rewards[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.as_tensor(Gt, dtype=torch.float32, device=self.device)
        
        # 添加奖励数据到TensorBoard
        if rewards:
            avg_total_reward = sum(rewards) / len(rewards)
            tb.add_scalar('Reward/Total_Avg', avg_total_reward, i_ep)
            tb.add_scalar('Reward/Extrinsic_Avg', avg_total_reward - avg_intrinsic_reward, i_ep)

        # PPO更新
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(modified_buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('Epoch {} , Training step {}'.format(i_ep, self.training_step))
                    
                # 准备批量数据
                states_batch = state[index]
                actions_batch = action[index]
                old_probs_batch = old_action_log_prob[index]
                Gt_batch = Gt[index].view(-1, 1)
                
                # 价值预测
                V = self.critic_net(states_batch)
                delta = Gt_batch - V
                advantage = delta.detach()
                
                # 动作概率计算
                action_probs = self.actor_net(states_batch)
                selected_action_probs = action_probs.gather(1, actions_batch)
                ratio = (selected_action_probs / old_probs_batch)
                
                # PPO损失计算
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # 更新actor网络
                action_loss = -torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                self.action_loss = action_loss.item()

                # 更新critic网络
                value_loss = F.mse_loss(Gt_batch, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.value_loss = value_loss.item()
                
                self.training_step += 1

        # 清空经验池
        self.buffer.clear()

# 主函数
def main():
    agent = PPO()
    rospy.init_node('turtlebot3_dqn_stage_4')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    start_time = time.time()
    
    # 训练循环
    for e in range(300):
        state = env.reset()
        episode_reward_sum = 0
        episode_intrinsic_reward = 0
        done = False
        episode_step = 6000
        episode_actions = []

        # 回合内步进循环
        for t in range(episode_step):
            # 选择动作
            action, action_prob = agent.select_action(state)
            next_state, extrinsic_reward, done = env.step(action)
            
            # 计算内在奖励
            intrinsic_reward = agent.icm.get_intrinsic_reward(state, next_state, action)
            total_reward = extrinsic_reward + intrinsic_reward
            
            # 存储经验
            trans = Transition(state, action, action_prob, total_reward, next_state)
            agent.store_transition(trans)
            
            # 更新状态和奖励
            state = next_state
            episode_reward_sum += total_reward
            episode_intrinsic_reward += intrinsic_reward
            episode_actions.append(action)
            
            # 每10回合保存一次模型
            if e % 10 == 0:
                agent.save_param(e)
                
            # 超时处理
            if t >= 600:
                rospy.loginfo("Time out!")
                done = True

            # 回合结束处理
            if done:
                # 记录结果
                result.data = [
                    episode_reward_sum, 
                    agent.action_loss, 
                    agent.value_loss,
                    episode_intrinsic_reward,
                    np.mean(episode_actions) if episode_actions else 0
                ]
                pub_result.publish(result)
                
                # TensorBoard记录
                tb.add_scalar('Episode/Reward_Total', episode_reward_sum, e)
                tb.add_scalar('Episode/Reward_Intrinsic', episode_intrinsic_reward, e)
                tb.add_scalar('Episode/Reward_Extrinsic', episode_reward_sum - episode_intrinsic_reward, e)
                tb.add_scalar('Loss/Action', agent.action_loss, e)
                tb.add_scalar('Loss/Value', agent.value_loss, e)
                tb.add_scalar('Info/Actions_Avg', np.mean(episode_actions), e)
                
                # 计算并记录时间信息
                elapsed_time = time.time() - start_time
                m, s = divmod(int(elapsed_time), 60)
                h, m = divmod(m, 60)
                
                # 打印回合信息
                rospy.loginfo(
                    'Ep: %d TotalReward: %.2f Intrinsic: %.2f Extrinsic: %.2f Steps: %d Time: %d:%02d:%02d', 
                    e, episode_reward_sum, episode_intrinsic_reward, 
                    episode_reward_sum - episode_intrinsic_reward, t, h, m, s
                )
                
                # 更新策略
                agent.update(e)
                break

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    print("Training complete")