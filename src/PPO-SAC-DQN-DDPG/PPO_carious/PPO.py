#!/usr/bin/env python
# coding=UTF-8

from collections import namedtuple
from itertools import count
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from environment_stage_4 import Env
import rospy
from std_msgs.msg import Float32MultiArray

tb = SummaryWriter('/home/dell/sztu/RL_nav/src/ROS_pytorch_RL/PPO/runs')
# Parameters
gamma = 0.99
render = False
seed = 1
log_interval = 10

# Environment dimensions from environment_stage_4.py
num_state = 28  # 24 lidar sectors + heading + current distance + min obstacle distance + obstacle sector
num_action = 5  # 5 discrete actions
env = Env(num_action)
torch.manual_seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.state_value = nn.Linear(100, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = self.state_value(x)
        return value

# ICM Network for Curiosity-driven exploration
class ICM(nn.Module):
    def __init__(self, state_dim=num_state, action_dim=num_action, hidden_dim=128, beta=0.2):
        super(ICM, self).__init__()
        self.beta = beta
        
        # Feature network: state -> feature vector
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Inverse model: (current state, next state) -> action prediction
        self.inverse_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Forward model: (current state feature, action) -> next state feature
        self.forward_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def compute_intrinsic_reward(self, state, next_state, action):
        # Convert to tensors if needed
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(next_state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long)
            
        state = state.unsqueeze(0) if len(state.shape) == 1 else state
        next_state = next_state.unsqueeze(0) if len(next_state.shape) == 1 else next_state
        
        with torch.no_grad():
            phi = self.feature_net(state)
            next_phi = self.feature_net(next_state)
            
            # Forward prediction
            action_onehot = F.one_hot(action, num_action).float()
            if len(action_onehot.shape) == 1:
                action_onehot = action_onehot.unsqueeze(0)
                
            input_forward = torch.cat([phi, action_onehot], dim=1)
            pred_next_phi = self.forward_net(input_forward)
            
            # Intrinsic reward: prediction error
            intrinsic_reward = F.mse_loss(pred_next_phi, next_phi, reduction='none').mean(dim=1)
            return intrinsic_reward.item()
    
    def update(self, states, actions, next_states):
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        
        # Compute features
        phi = self.feature_net(states)
        next_phi = self.feature_net(next_states)
        
        # Inverse model loss: action prediction
        inverse_input = torch.cat([phi, next_phi], dim=1)
        pred_actions_logits = self.inverse_net(inverse_input)
        inverse_loss = F.cross_entropy(pred_actions_logits, actions)
        
        # Forward model loss: next state prediction
        action_onehot = F.one_hot(actions, num_action).float()
        forward_input = torch.cat([phi, action_onehot], dim=1)
        pred_next_phi = self.forward_net(forward_input)
        forward_loss = F.mse_loss(pred_next_phi, next_phi)
        
        # Combined loss
        loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_optimizer(self, lr=1e-3):
        self.optimizer = optim.Adam(
            list(self.feature_net.parameters()) + 
            list(self.inverse_net.parameters()) + 
            list(self.forward_net.parameters()), 
            lr=lr
        )

class PPO:
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 10
    buffer_capacity = 1000
    batch_size = 128
    intrinsic_reward_scale = 0.01  # Scale factor for intrinsic reward

    def __init__(self):
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.icm = ICM()  # Curiosity module
        self.icm.get_optimizer(lr=1e-3)
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.action_loss = 0.
        self.value_loss = 0.
        self.load_models = False
        self.load_ep = 104
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-3)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-3)
        
        if self.load_models:
            load_model1 = torch.load("/home/dell/sztu/RL_nav/src/ROS_pytorch_RL/PPO/model260ppo.pt")
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
        state = {
            'actor_net': self.actor_net.state_dict(),
            'critic_net': self.critic_net.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_net_optimizer,
            'epoch': e
        }
        torch.save(state, "/home/dell/sztu/RL_nav/src/ROS_pytorch_RL/PPO/model" + str(e) + "ppo.pt")

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        # Extract data from buffer
        states = [t.state for t in self.buffer]
        actions = [t.action for t in self.buffer]
        rewards = [t.reward for t in self.buffer]
        old_action_log_probs = [t.a_log_prob for t in self.buffer]
        next_states = [t.next_state for t in self.buffer] if hasattr(self.buffer[0], 'next_state') else None

        # Calculate intrinsic rewards using ICM
        intrinsic_rewards = []
        for i in range(len(self.buffer)):
            # For the last transition, use previous next_state
            if i < len(self.buffer) - 1:
                next_state = self.buffer[i+1].state
            else:
                next_state = next_states[i] if next_states else states[i]
                
            intrinsic_reward = self.icm.compute_intrinsic_reward(
                states[i], 
                next_state, 
                actions[i]
            )
            intrinsic_rewards.append(intrinsic_reward)
        
        # Combine extrinsic and intrinsic rewards
        total_rewards = [r + self.intrinsic_reward_scale * i_r 
                         for r, i_r in zip(rewards, intrinsic_rewards)]
        
        # Convert to tensors
        state_tensor = torch.tensor(states, dtype=torch.float32)
        action_tensor = torch.tensor(actions, dtype=torch.long).view(-1, 1)
        old_action_log_prob_tensor = torch.tensor(old_action_log_probs, dtype=torch.float32).view(-1, 1)
        
        # Compute discounted rewards
        R = 0
        Gt = []
        for r in total_rewards[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt_tensor = torch.tensor(Gt, dtype=torch.float32)
        
        # PPO updates
        for _ in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                if self.training_step % 1000 == 0:
                    print(f'I_ep {i_ep}, train {self.training_step} times')
                
                # Compute advantages
                V = self.critic_net(state_tensor[index])
                delta = Gt_tensor[index] - V.squeeze()
                advantage = delta.detach()
                
                # Update actor
                action_prob = self.actor_net(state_tensor[index]).gather(1, action_tensor[index])
                ratio = action_prob / old_action_log_prob_tensor[index]
                surr1 = ratio * advantage.unsqueeze(1)
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage.unsqueeze(1)
                
                action_loss = -torch.min(surr1, surr2).mean()
                self.action_loss = action_loss.item()
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()
                
                # Update critic
                value_loss = F.mse_loss(Gt_tensor[index], V.squeeze())
                self.value_loss = value_loss.item()
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                
                self.training_step += 1
        
        # Update ICM using current buffer data
        if len(self.buffer) > 0 and next_states is not None and len(next_states) > 0:
            icm_loss = self.icm.update(states, actions, next_states)
            tb.add_scalar('ICM/Loss', icm_loss, i_ep)
        
        # Clear buffer
        del self.buffer[:]


def main():
    print("Start PPO training")
    agent = PPO()
    rospy.init_node('turtlebot3_dqn_stage_4')
    print("Start PPO training")
    pub_result = rospy.Publisher('result', Float32MultiArray, queue_size=5)
    pub_get_action = rospy.Publisher('get_action', Float32MultiArray, queue_size=5)
    result = Float32MultiArray()
    get_action = Float32MultiArray()
    start_time = time.time()
    
    for e in range(6000):
        state = env.reset()
        episode_reward_sum = 0
        done = False
        episode_step = 6000

        for t in range(episode_step):
            action, action_prob = agent.select_action(state)
            next_state, reward, done = env.step(action)
            
            # Store transition with next_state for ICM
            trans = Transition(state, action, action_prob, reward, next_state)
            agent.store_transition(trans)
            
            state = next_state
            episode_reward_sum += reward
            pub_get_action.publish(get_action)
            
            if e % 30 == 0:
                agent.save_param(e)
                
            if t >= 600:
                rospy.loginfo("time out!")
                done = True
                
            if done:
                result.data = [episode_reward_sum, agent.action_loss, agent.value_loss]
                pub_result.publish(result)
                tb.add_scalar('Loss', episode_reward_sum, e)
                tb.add_scalar('value_loss', agent.value_loss, e)
                tb.add_scalar('action_loss', agent.action_loss, e)
                
                # Update agent with the episode's experiences
                agent.update(e)
                
                # Calculate elapsed time correctly
                elapsed_time = time.time() - start_time
                hours, remainder = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                
                rospy.loginfo('Ep: %d score: %.2f memory: %d episode_step: %d time: %d:%02d:%02d',
                              e, episode_reward_sum, agent.counter, t,
                              int(hours), int(minutes), int(seconds))
                break
                
    tb.close()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    print("end")
