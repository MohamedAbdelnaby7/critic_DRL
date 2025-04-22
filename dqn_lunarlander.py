#!/usr/bin/env python3
"""
DQN implementation for LunarLander-v2 (discrete action space).
Usage: python dqn_lunarlander.py
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
from torch.utils.tensorboard import SummaryWriter
import time

# ---------------------
# Hyperparameters TUNING
# Adjust as needed
# ---------------------
BATCH_SIZE = 64
LR = 1e-3
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 50000  # Steps over which eps decays
TARGET_UPDATE_FREQ = 1000
MEMORY_CAPACITY = 100000
MAX_EPISODES = 1000
MAX_STEPS = 500
RENDER = False

# ----- Neural Network for Q-value Approximation -----
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# ----- Replay Buffer -----
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

# ----- DQN Agent -----
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_q_net = QNetwork(state_dim, action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LR)
        self.buffer = ReplayBuffer(MEMORY_CAPACITY)
        self.global_step = 0

    def select_action(self, state, epsilon):
        # Epsilon-greedy
        if random.random() < epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_v = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_net(state_v)
                _, action = torch.max(q_values, dim=1)
                return int(action.item())

    def update(self, writer=None):
        if len(self.buffer) < BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        
        states_v = torch.FloatTensor(states)
        actions_v = torch.LongTensor(actions).unsqueeze(-1)
        rewards_v = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states_v = torch.FloatTensor(next_states)
        dones_v = torch.FloatTensor(dones).unsqueeze(-1)

        # Current Q estimates
        q_values = self.q_net(states_v).gather(1, actions_v)

        # Next Q (from target network)
        with torch.no_grad():
            max_next_q = self.target_q_net(next_states_v).max(1, keepdim=True)[0]
            target_q = rewards_v + (1 - dones_v) * GAMMA * max_next_q

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optimizer.step()

        if writer is not None:
            writer.add_scalar("loss/td_loss", loss.item(), self.global_step)

        # Periodically update target network
        if self.global_step % TARGET_UPDATE_FREQ == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

    def store_transition(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

# -------------- MAIN TRAINING LOOP --------------

def main():
    env = gym.make("LunarLander-v3", render_mode="human" if RENDER else None)
    # if needed: env = gym.wrappers.RecordEpisodeStatistics(env)

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(obs_dim, action_dim)
    writer = SummaryWriter(comment="_DQN_LunarLander")
    print("Beginning DQN training on LunarLander-v2...")

    epsilon = EPS_START
    epsilon_decay = (EPS_START - EPS_END) / EPS_DECAY
    global_step = 0
    best_avg_reward = -9999

    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(MAX_STEPS):
            global_step += 1
            agent.global_step = global_step

            action = agent.select_action(state, epsilon)
            next_state, reward, done, truncated, info = env.step(action)
            done_bool = done or truncated

            agent.store_transition(state, action, reward, next_state, done_bool)
            agent.update(writer)

            state = next_state
            episode_reward += reward

            # Epsilon Decay
            if epsilon > EPS_END:
                epsilon -= epsilon_decay
            
            if done_bool:
                break
        
        writer.add_scalar("reward/episode", episode_reward, episode)
        # evaluate
        print(f"Episode={episode}, Reward={episode_reward:.2f}, Epsilon={epsilon:.3f}")
    
    env.close()
    writer.close()

if __name__ == "__main__":
    main()

