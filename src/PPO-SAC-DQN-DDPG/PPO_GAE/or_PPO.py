#!/usr/bin/env python
# coding=UTF-8

from collections import namedtuple
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
tb =SummaryWriter('/home/dell/drl/src/PPO-SAC-DQN-DDPG/PPO/runs')
# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10


# env = gym.make('CartPole-v0').unwrapped
# action个数为19，observation为115。
num_state =28
num_action = 5
env=Env(num_action)
torch.manual_seed(seed)#为CPU设置种子用于生成随机数，以使得结果是确定的
# env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action',  'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):#Actor网络 
    def __init__(self):#定义网络
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        # self.fc1.weight.data.normal_(0, 0.1)
        # self.fc2 =nn.Linear(128,128)
        # self.fc2.weight.data.normal_(0,0.1)
        self.action_head = nn.Linear(100, num_action)
        # self.action_head.weight.data.normal_(0, 0.1)  

    def forward(self, x):#前向传播
        x = F.relu(self.fc1(x))
        # x=F.relu(self.fc2(x))
        # x=F.dropout(self.fc2(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):#Critic网络
    def __init__(self):#定义网络
        super(Critic, self).__init__()
        self.fc1= nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)
        
    def forward(self, x):#前向传播
        x = F.relu(self.fc1(x))
        # x=F.dropout(self.fc22(x))
        value = self.state_value(x)
        return value


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
        self.action_loss= 0.
        self.value_loss =0.
        self.load_models =False
        self.load_ep =104
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        # Adam(Adaptive Moment Estimation)本质上是带有动量项的RMSprop，它利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。它的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。
        #加载模型
        if self.load_models:
            load_model1 = torch.load("/home/ffd/DRL/PPO/model/maze/98ep.pt")
            self.actor_net.load_state_dict(load_model1['actor_net'])
            self.critic_net.load_state_dict(load_model1['critic_net'])
            print("load model:",str(self.load_ep))
            print("load model successful!!!!!!")
#选择动作
    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0) 
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self,e):
        state = {'actor_net':self.actor_net.state_dict(),'critic_net':self.critic_net.state_dict(), 'actor_optimizer':self.actor_optimizer.state_dict(), 'critic_optimizer':self.critic_net_optimizer,'epoch':e}
        torch.save(state,"/home/dell/drl/src/PPO-SAC-DQN-DDPG/PPO/model/"+str(e)+"ppo.pt")

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    # PPO-V1
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


    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float)
        #print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 ==0:
                    print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!一次训练的参数更新
                action_prob = self.actor_net(state[index]).gather(1, action[index]) # new policy
                #采用 Adam 随机梯度上升算法最大化 PPO-Clip 的目标函数来更新策略
                #
                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.action_loss = torch.max(action_loss)
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.value_loss = torch.max(value_loss)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience


def main():
    agent = PPO()
    rospy.init_node('turtlebot3_dqn_stage_4')
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    start_time =time.time()
    # env=Env()
    for e in range(300):
        state = env.reset()#env.reset()函数用于重置环境
        # if render: env.render()#env.render()函数用于渲染出当前的智能体以及环境的状态
        episode_reward_sum = 0                                              # 初始化该循环对应的episode的总奖励
        done=False
        episode_step=6000

        for t in range(episode_step):
            action, action_prob = agent.select_action(state)
            next_state, reward, done= env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            # if render: env.render()
            agent.store_transition(trans)
            state = next_state
            episode_reward_sum+=reward
            pub_get_action.publish(get_action)
            if e % 10 ==0:                # dqn.save_model(str(e))
                agent.save_param(e)
            if t >=600:
                rospy.loginfo("time out!")
                done =True

            if done :
                result.data =[episode_reward_sum,agent.action_loss,agent.value_loss]
                pub_result.publish(result)
                tb.add_scalar('Loss',  episode_reward_sum,e)
                tb.add_scalar('value_loss',agent.value_loss, e)
                tb.add_scalar('action_loss', agent.action_loss, e)

                m,s =divmod(int(time.time()- start_time),60)
                h,m =divmod(m,60)
                agent.update(e)
                rospy.loginfo('Ep: %d score: %.2f memory: %d episode_step: %.2f time: %d:%02d:%02d' , e ,episode_reward_sum, agent.counter,t, h, m, s)
                break
if __name__ == '__main__':

    main()
    print("end")

