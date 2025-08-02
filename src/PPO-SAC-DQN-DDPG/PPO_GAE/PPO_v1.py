#!/usr/bin/env python
# coding=UTF-8

from collections import namedtuple, deque
from itertools import count
import os, time
import numpy as np
# import matplotlib.pyplot as plt

# import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from environment_stage_4_ppo import Env
import time
import rospy
import tensorboard
from std_msgs.msg import Float32MultiArray
from utils import make_mlp, GAE

# 创建TensorBoard写入器
tb = SummaryWriter('/home/dell/drl/src/PPO-SAC-DQN-DDPG/PPO/runs')

# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10

# env = gym.make('CartPole-v0').unwrapped
# action个数为19，observation为115。
num_state = 28
num_action = 5
env = Env(num_action)
torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
# env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

class RewardNormalizer:
    """奖励归一化类，跟踪奖励的统计信息并进行归一化"""
    def __init__(self, epsilon=1e-8, cliprew=10.0):
        self.running_mean = 0
        self.running_std = 0
        self.count = epsilon
        self.epsilon = epsilon
        self.cliprew = cliprew

    def normalize(self, reward):
        # 计算奖励的均值和标准差
        self.count += 1
        delta = reward - self.running_mean
        self.running_mean += delta / self.count
        self.running_std = np.sqrt((self.running_std**2 * (self.count - 1) + delta**2) / self.count)
        
        # 归一化并裁剪奖励
        norm_reward = reward / (self.running_std + self.epsilon)
        return np.clip(norm_reward, -self.cliprew, self.cliprew)

    def get_stats(self):
        return self.running_mean, self.running_std

# class Actor(nn.Module):  # Actor网络
#     def __init__(self):  # 定义网络
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(num_state, 100)
#         self.action_head = nn.Linear(100, num_action)

#     def forward(self, x):  # 前向传播
#         x = F.relu(self.fc1(x))
#         action_prob = F.softmax(self.action_head(x), dim=1)
#         return action_prob


# class Critic(nn.Module):  # Critic网络
#     def __init__(self):  # 定义网络
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(num_state, 100)
#         self.state_value = nn.Linear(100, 1)
        
#     def forward(self, x):  # 前向传播
#         x = F.relu(self.fc1(x))
#         value = self.state_value(x)
#         return value


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.net = make_mlp([100, num_action])

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=1)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.net = make_mlp([100, 1])

    def forward(self, x):
        return self.net(x)


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
        self.gae = GAE(gamma=0.99, lmbda=0.95)
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        
        # 奖励归一化器
        self.reward_normalizer = RewardNormalizer()
        
        # 损失平滑器
        self.action_loss_smoother = ExponentialSmoother(alpha=0.95)
        self.value_loss_smoother = ExponentialSmoother(alpha=0.95)
        
        # 加载模型
        if self.load_models:
            load_model1 = torch.load("/home/ffd/DRL/PPO/model/maze/98ep.pt")
            self.actor_net.load_state_dict(load_model1['actor_net'])
            self.critic_net.load_state_dict(load_model1['critic_net'])
            print("load model:", str(self.load_ep))
            print("load model successful!!!!!!")

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0) 
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self, e):
        state = {'actor_net': self.actor_net.state_dict(), 'critic_net': self.critic_net.state_dict(), 
                 'actor_optimizer': self.actor_optimizer.state_dict(), 'critic_optimizer': self.critic_net_optimizer, 'epoch': e}
        torch.save(state, "/home/dell/drl/src/PPO-SAC-DQN-DDPG/PPO/model/" + str(e) + "ppo.pt")

    def store_transition(self, transition):
        # 归一化奖励
        norm_reward = self.reward_normalizer.normalize(transition.reward)
        norm_transition = Transition(transition.state, transition.action, 
                                    transition.a_log_prob, norm_reward, transition.next_state)
        self.buffer.append(norm_transition)
        self.counter += 1

    # def update(self, i_ep, episode_rewards):
    #     # 优化：先转换为NumPy数组，再创建张量
    #     states = np.array([t.state for t in self.buffer])
    #     actions = np.array([t.action for t in self.buffer])
    #     rewards = np.array([t.reward for t in self.buffer])
    #     old_action_log_probs = np.array([t.a_log_prob for t in self.buffer])
        
    #     # 转换为张量
    #     state = torch.tensor(states, dtype=torch.float)
    #     action = torch.tensor(actions, dtype=torch.long).view(-1, 1)
    #     reward = [t.reward for t in self.buffer]  # 保持原列表用于后续计算
    #     old_action_log_prob = torch.tensor(old_action_log_probs, dtype=torch.float).view(-1, 1)


    #     # 使用GT的方法计算损失
    #     # R = 0
    #     # Gt = []
    #     # for r in reward[::-1]:
    #     #     R = r + gamma * R
    #     #     Gt.insert(0, R)
    #     # Gt = torch.tensor(Gt, dtype=torch.float)
        
    #     # 使用 GAE计算损失
    #     gae = GAE(gamma=0.99, lmbda=0.95)  # 可放在 PPO 初始化中缓存
    #     with torch.no_grad():
    #         values = self.critic_net(state).squeeze(-1)  # shape (batch,)
    #         next_state_tensor = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
    #         next_values = self.critic_net(next_state_tensor).squeeze(-1)  # shape (batch,)
            
    #     # reshape to (1, batch) to match GAE expected input: (B, T)
    #     rewards = torch.tensor(rewards).unsqueeze(0)
    #     dones = torch.zeros_like(rewards)  # 无明确终止标签，设置为0
    #     values = values.unsqueeze(0)
    #     next_values = next_values.unsqueeze(0)

    #     adv, Gt = gae(rewards, dones, values, next_values)
    #     Gt = Gt.squeeze(0).detach()
    #     adv = adv.squeeze(0).detach()

    #     # 记录奖励统计信息
    #     tb.add_scalar('Reward/Mean', np.mean(episode_rewards), i_ep)
    #     tb.add_scalar('Reward/Max', np.max(episode_rewards), i_ep)
    #     tb.add_scalar('Reward/Min', np.min(episode_rewards), i_ep)
    #     tb.add_scalar('Reward/Std', np.std(episode_rewards), i_ep)
        
    #     # 记录奖励归一化统计信息
    #     reward_mean, reward_std = self.reward_normalizer.get_stats()
    #     tb.add_scalar('Reward/Normalized_Mean', reward_mean, i_ep)
    #     tb.add_scalar('Reward/Normalized_Std', reward_std, i_ep)

    #     #print("The agent is updateing....")
    #     action_losses = []
    #     value_losses = []
        
    #     for i in range(self.ppo_update_time):
    #         for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
    #             if self.training_step % 1000 == 0:
    #                 print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                
    #             Gt_index = Gt[index].view(-1, 1)
    #             V = self.critic_net(state[index])
    #             delta = Gt_index - V
    #             advantage = delta.detach()
                
    #             # epoch iteration, PPO core!!一次训练的参数更新
    #             action_prob = self.actor_net(state[index]).gather(1, action[index])  # new policy
                
    #             ratio = (action_prob / old_action_log_prob[index])
    #             surr1 = ratio * advantage
    #             surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

    #             # update actor network
    #             action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
    #             action_losses.append(action_loss.item())
    #             self.action_loss = torch.max(action_loss)
    #             self.actor_optimizer.zero_grad()
    #             action_loss.backward()
    #             nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
    #             self.actor_optimizer.step()

    #             # update critic network
    #             value_loss = F.mse_loss(Gt_index, V)
    #             value_losses.append(value_loss.item())
    #             self.value_loss = torch.max(value_loss)
    #             self.critic_net_optimizer.zero_grad()
    #             value_loss.backward()
    #             nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
    #             self.critic_net_optimizer.step()
    #             self.training_step += 1

    #     # 计算平均损失并平滑
    #     mean_action_loss = np.mean(action_losses)
    #     mean_value_loss = np.mean(value_losses)
        
    #     # 应用指数平滑
    #     smoothed_action_loss = self.action_loss_smoother.update(mean_action_loss)
    #     smoothed_value_loss = self.value_loss_smoother.update(mean_value_loss)
        
    #     # 记录原始损失和平滑损失
    #     tb.add_scalar('Loss/Action_Loss', mean_action_loss, i_ep)
    #     tb.add_scalar('Loss/Value_Loss', mean_value_loss, i_ep)
    #     tb.add_scalar('Loss/Smoothed_Action_Loss', smoothed_action_loss, i_ep)
    #     tb.add_scalar('Loss/Smoothed_Action_Loss', smoothed_value_loss, i_ep)
        
    #     # 记录损失的标准差
    #     tb.add_scalar('Loss/Action_Loss_Std', np.std(action_losses), i_ep)
    #     tb.add_scalar('Loss/Value_Loss_Std', np.std(value_losses), i_ep)

    #     del self.buffer[:]  # clear experience


    def update(self, i_ep, episode_rewards):
        states = np.array([t.state for t in self.buffer], dtype=np.float32)
        actions = np.array([t.action for t in self.buffer])
        rewards = np.array([t.reward for t in self.buffer], dtype=np.float32)
        old_action_log_probs = np.array([t.a_log_prob for t in self.buffer], dtype=np.float32)
        next_states = np.array([t.next_state for t in self.buffer], dtype=np.float32)

        state = torch.tensor(states)  # (B, state_dim)
        next_state_tensor = torch.tensor(next_states)
        action = torch.tensor(actions, dtype=torch.long).view(-1, 1)
        old_action_log_prob = torch.tensor(old_action_log_probs).view(-1, 1)

        # -------- GAE计算 --------
        with torch.no_grad():
            values = self.critic_net(state).squeeze(-1)  # (B,)
            next_values = self.critic_net(next_state_tensor).squeeze(-1)  # (B,)

        # 将 reward, done, value, next_value 改成 batch of sequence (1, B)
        rewards_tensor = torch.tensor(rewards).unsqueeze(0)
        terminated = torch.zeros_like(rewards_tensor)  # 无终止标记，设为0
        values = values.unsqueeze(0)
        next_values = next_values.unsqueeze(0)

        adv, Gt = self.gae(rewards_tensor, terminated, values, next_values)
        adv = adv.squeeze(0).detach()
        Gt = Gt.squeeze(0).detach()

        # -------- 记录统计 --------
        tb.add_scalar('Reward/Mean', np.mean(episode_rewards), i_ep)
        tb.add_scalar('Reward/Max', np.max(episode_rewards), i_ep)
        tb.add_scalar('Reward/Min', np.min(episode_rewards), i_ep)
        tb.add_scalar('Reward/Std', np.std(episode_rewards), i_ep)

        reward_mean, reward_std = self.reward_normalizer.get_stats()
        tb.add_scalar('Reward/Normalized_Mean', reward_mean, i_ep)
        tb.add_scalar('Reward/Normalized_Std', reward_std, i_ep)

        action_losses = []
        value_losses = []

        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print('I_ep {} ，train {} times'.format(i_ep, self.training_step))

                V = self.critic_net(state[index])
                advantage = adv[index].unsqueeze(-1)

                # ------- Actor 更新 -------
                new_action_prob = self.actor_net(state[index]).gather(1, action[index])
                ratio = new_action_prob / old_action_log_prob[index]
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
                action_loss = -torch.min(surr1, surr2).mean()
                action_losses.append(action_loss.item())

                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # ------- Critic 更新 -------
                value_loss = F.mse_loss(Gt[index].unsqueeze(-1), V)
                value_losses.append(value_loss.item())

                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

                self.training_step += 1

        # 记录损失
        tb.add_scalar('Loss/Action_Loss', np.mean(action_losses), i_ep)
        tb.add_scalar('Loss/Value_Loss', np.mean(value_losses), i_ep)
        tb.add_scalar('Loss/Action_Loss_Std', np.std(action_losses), i_ep)
        tb.add_scalar('Loss/Value_Loss_Std', np.std(value_losses), i_ep)

        del self.buffer[:]



class ExponentialSmoother:
    """指数平滑器，用于平滑曲线"""
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.value = None
        
    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * new_value
        return self.value

def main():
    agent = PPO()
    rospy.init_node('turtlebot3_dqn_stage_4')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    start_time = time.time()
    
    # 用于存储所有episode的奖励
    all_episode_rewards = []
    
    for e in range(1200):
        state = env.reset()  # env.reset()函数用于重置环境
        # if render: env.render()  # env.render()函数用于渲染出当前的智能体以及环境的状态
        episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励
        done = False
        episode_step = 6000
        episode_rewards = []  # 存储当前episode的所有奖励
        
        for t in range(episode_step):
            action, action_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            # if render: env.render()
            agent.store_transition(trans)
            state = next_state
            episode_reward_sum += reward
            episode_rewards.append(reward)
            pub_get_action.publish(get_action)
            
            if e % 10 == 0:  # dqn.save_model(str(e))
                agent.save_param(e)
                
            if t >= 600:
                rospy.loginfo("time out!")
                done = True

            if done:
                all_episode_rewards.append(episode_reward_sum)
                result.data = [episode_reward_sum, agent.action_loss, agent.value_loss]
                pub_result.publish(result)
                
                # 记录原始奖励
                tb.add_scalar('Reward/Raw', episode_reward_sum, e)
                
                # 记录奖励的移动平均
                if len(all_episode_rewards) >= 10:
                    moving_avg = np.mean(all_episode_rewards[-10:])
                    tb.add_scalar('Reward/Moving_Average_10', moving_avg, e)
                
                m, s = divmod(int(time.time() - start_time), 60)
                h, m = divmod(m, 60)
                agent.update(e, episode_rewards)
                rospy.loginfo('Ep: %d score: %.2f memory: %d episode_step: %.2f time: %d:%02d:%02d', 
                             e, episode_reward_sum, agent.counter, t, h, m, s)
                break
                
    # 训练结束后记录最终统计信息
    tb.add_hparams(
        {'gamma': gamma, 'clip_param': PPO.clip_param, 'lr_actor': 1e-3, 'lr_critic': 3e-3},
        {
            'hparam/avg_reward': np.mean(all_episode_rewards),
            'hparam/max_reward': np.max(all_episode_rewards),
            'hparam/min_reward': np.min(all_episode_rewards),
            'hparam/std_reward': np.std(all_episode_rewards)
        }
    )

if __name__ == '__main__':
    main()
    print("end")