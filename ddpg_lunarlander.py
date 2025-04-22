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
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.95
TAU = 0.005  # Soft update
MEMORY_CAPACITY = 50000
BATCH_SIZE = 64
NOISE_SCALE_INIT = 0.1
NOISE_SCALE_END = 0.01
MAX_EPISODES = 1000
MAX_STEPS = 1000
RENDER = False

# ---------------------
# CUDA Device Setup
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----- Actor Network -----
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 512),  # Increased number of neurons
            nn.ReLU(),
            nn.LayerNorm(512),  # Use Layer Normalization instead of BatchNorm
            nn.Linear(512, 512),  # Increased layer width
            nn.ReLU(),
            nn.LayerNorm(512),  # Use Layer Normalization instead of BatchNorm
            nn.Linear(512, action_dim),
            nn.Tanh()  # Output bounded between [-1, 1] for continuous action
        )
        self.max_action = max_action

    def forward(self, x):
        x = self.net(x)
        # use tanh for bounding
        return self.max_action * torch.tanh(x)

# ----- Critic Network -----
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 512),  # Increased number of neurons
            nn.ReLU(),
            nn.BatchNorm1d(512),  # Batch normalization
            nn.Linear(512, 512),  # Increased layer width
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

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
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

# ----- DDPG Agent -----
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

        self.buffer = ReplayBuffer(MEMORY_CAPACITY)
        self.global_step = 0

    def select_action(self, state, noise_scale):
        with torch.no_grad():
            state_v = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = self.actor(state_v).cpu().numpy().flatten()
        # Add gaussian noise for exploration
        noise = np.random.normal(0, noise_scale, size=self.action_dim)
        action = action + noise
        return np.clip(action, -self.max_action, self.max_action)

    def store_transition(self, s, a, r, s_next, done):
        self.buffer.push(s, a, r, s_next, done)

    def update(self, writer=None):
        if len(self.buffer) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        states_v = torch.FloatTensor(states).to(device)
        actions_v = torch.FloatTensor(actions).to(device)
        rewards_v = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
        next_states_v = torch.FloatTensor(next_states).to(device)
        dones_v = torch.FloatTensor(dones).unsqueeze(-1).to(device)

        # Critic update
        with torch.no_grad():
            next_actions = self.actor_target(next_states_v)
            target_q = self.critic_target(next_states_v, next_actions)
            y = rewards_v + (1 - dones_v) * GAMMA * target_q
        
        q_val = self.critic(states_v, actions_v)
        critic_loss = nn.MSELoss()(q_val, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_opt.step()

        # Actor update
        pred_actions = self.actor(states_v)
        actor_loss = -self.critic(states_v, pred_actions).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_opt.step()

        # soft update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        if writer is not None:
            step = self.global_step
            writer.add_scalar("loss/critic", critic_loss.item(), step)
            writer.add_scalar("loss/actor", actor_loss.item(), step)

def main():
    env = gym.make("LunarLanderContinuous-v3", render_mode="human" if RENDER else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])  # for LunarLanderContinuous is 1.0

    agent = DDPGAgent(state_dim, action_dim, max_action)
    writer = SummaryWriter(comment="_DDPG_LunarLander")

    print("Beginning DDPG training on LunarLanderContinuous-v2...")

    noise_decay_steps = 50000
    noise_scale = NOISE_SCALE_INIT
    noise_decay = (NOISE_SCALE_INIT - NOISE_SCALE_END) / noise_decay_steps

    global_step = 0
    for episode in range(MAX_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        for t in range(MAX_STEPS):
            global_step += 1
            agent.global_step = global_step
            action = agent.select_action(state, noise_scale)
            next_state, reward, done, truncated, info = env.step(action)
            done_bool = done or truncated

            agent.store_transition(state, action, reward, next_state, done_bool)
            agent.update(writer)

            episode_reward += reward
            state = next_state

            # noise scale decay
            if noise_scale > NOISE_SCALE_END:
                noise_scale -= noise_decay

            if done_bool:
                break

        writer.add_scalar("reward/episode", episode_reward, episode)
        print(f"Episode={episode}, Reward={episode_reward:.2f}, Noise={noise_scale:.3f}")
    
    env.close()
    writer.close()

if __name__ == "__main__":
    main()
