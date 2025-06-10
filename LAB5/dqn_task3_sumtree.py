# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        # An example: 
        #self.network = nn.Sequential(
        #    nn.Linear(input_dim, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, 64),
        #    nn.ReLU(),
        #    nn.Linear(64, num_actions)
        #)       
        ########## YOUR CODE HERE (5~10 lines) ##########
        ## task2
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        # task1
        # self.network = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(4, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, num_actions)
        # )
        
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        # return self.network(x)
        return self.network(x / 255.0)  # Normalize to 0~1


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 1:  # CartPole 是 1D
            return obs
        else:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        if len(frame.shape) == 1:
            return frame  # CartPole 直接回傳 state vector
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        if len(frame.shape) == 1:
            return frame # CartPole 直接回傳 state vector
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

###

class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4,reward_scale=1):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6
        self.reward_scale = reward_scale
        self.beta_increment_per_sampling = 0.001
        # self.buffer = []
        # self.priorities = np.zeros((capacity,), dtype=np.float32)
        # self.pos = 0

    def __len__(self):
        return self.tree.n_entries
    
    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
        # calculate the priority,error is TD error
        scaled_error = abs(error / self.reward_scale)
        priority = (scaled_error + self.epsilon) ** self.alpha        
        # add the transition to the tree
        self.tree.add(priority, transition)
            
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
        # sample the batch_size of transitions
        batch = []
        idxs = []
        priorities = []        
        segment = self.tree.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        
        for i in range(batch_size):
            s = random.uniform(i * segment, (i + 1) * segment)
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        # importance sampling weights
        sampling_probs = np.array(priorities) / self.tree.total()
        weights = np.power(self.tree.n_entries * sampling_probs , -self.beta)
        weights /= weights.max()
        
        batch = list(zip(*batch))
        states, actions, rewards, next_states, dones = batch
        
        states = np.array(batch[0])
        actions = torch.tensor(batch[1], dtype=torch.int64)
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = np.array(batch[3])
        dones = torch.tensor(batch[4], dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        ########## END OF YOUR CODE (for Task 3) ########## 
        return states, actions, rewards, next_states, dones, weights, idxs

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        # update the priorities of the sampled transitions
        for i, idx in enumerate(indices):
            scaled_error = abs(errors[i] / self.reward_scale)
            priority = (scaled_error + self.epsilon) ** self.alpha
            self.tree.update(idx, priority) 
        ########## END OF YOUR CODE (for Task 3) ########## 
        # return
        

class DQNAgent:         # task1 CartPole-v1  # task2 ALE/Pong-v5
    def __init__(self, env_name="ALE/Pong-v5", args=None): 
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)


        self.q_net = DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = os.path.join(
            args.save_dir,
            f"{args.wandb_run_name}_{time.strftime('%Y%m%d-%H%M%S')}",
            f"{env_name}"  # task1 CartPole-v1  # task2 ALE/Pong-v5
        )  
        os.makedirs(self.save_dir, exist_ok=True)

        # self.memory = deque(maxlen=args.memory_size)
        self.memory = PrioritizedReplayBuffer(args.memory_size, alpha=0.6, beta=0.4, reward_scale=args.reward_scale)
        
        self.n_step = getattr(args, "n_step", 2)
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.reward_scale = args.reward_scale
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                # task2 
                reward = np.clip(reward, -1, 1)  *self.reward_scale

                next_state = self.preprocessor.step(next_obs)
                
                # task2: direactly add the transition to the replay buffer
                # self.memory.append((state, action, reward, next_state, done))
                
                # task3: Compute TD error before add()
                # self.memory.add((state, action, reward, next_state, done), td_error.item())

                self.n_step_buffer.append((state, action, reward, next_state, done))
                if len(self.n_step_buffer) == self.n_step:
                    # Calculate the n-step return
                    R, S , A = 0, self.n_step_buffer[0][0], self.n_step_buffer[0][1]
                    D = False
                    for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                        R += (self.gamma ** i) * r
                        if d:
                            D = True
                            break
                    S_next = self.n_step_buffer[-1][3]
                    # Calculate the TD error
                    
                    with torch.no_grad():
                        s_tensor = torch.from_numpy(np.array(S)).float().unsqueeze(0).to(self.device)
                        s_next_tensor = torch.from_numpy(np.array(S_next)).float().unsqueeze(0).to(self.device)
                        q_val = self.q_net(s_tensor)[0][A].cpu()
                        max_next_q = self.target_net(s_next_tensor).max(1)[0].cpu()
                        td_error = R + (1 - D) * (self.gamma ** self.n_step) * max_next_q - q_val


                    # Add the transition to the replay buffer
                    self.memory.add((S, A, R, S_next, D), td_error.item())

                for _ in range(self.train_per_step):
                    self.train()
                
                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1
                

                if self.env_count % 1000 == 0:        
                    # snapshot frequency: 200k
                    if self.env_count % 200000 == 0 and self.env_count in [200000, 400000, 600000, 800000, 1000000]:
                        model_path = os.path.join(self.save_dir, f"LAB5_313554044_task3_pong{self.env_count}.pt")
                        torch.save(self.q_net.state_dict(), model_path)    

                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            
            # Flush 殘留的 n-step buffer（遞減 n）
            while len(self.n_step_buffer) > 0:
                R, S, A = 0, self.n_step_buffer[0][0], self.n_step_buffer[0][1]
                D = False
                for i, (_, _, r, _, d) in enumerate(self.n_step_buffer):
                    R += (self.gamma ** i) * r
                    if d:
                        D = True
                        break
                S_next = self.n_step_buffer[-1][3]

                with torch.no_grad():
                    s_tensor = torch.from_numpy(np.array(S)).float().unsqueeze(0).to(self.device)
                    s_next_tensor = torch.from_numpy(np.array(S_next)).float().unsqueeze(0).to(self.device)
                    q_val = self.q_net(s_tensor)[0][A].cpu()
                    max_next_q = self.target_net(s_next_tensor).max(1)[0].cpu()
                    td_error = R + (1 - D) * (self.gamma ** len(self.n_step_buffer)) * max_next_q - q_val

                self.memory.add((S, A, R, S_next, D), td_error.item())
                self.n_step_buffer.popleft() 

            
            # End of episode
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  

            if ep % 20 == 0:
                if ep % 100 == 0:
                    model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved model checkpoint to {model_path}")
                
                
                eval_reward = self.evaluate()
                if eval_reward >= self.best_reward and eval_reward > 19:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, f"best_model_{ep}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })
                
            # Decay function for epsilin-greedy exploration
            # if self.epsilon > self.epsilon_min:
            #     self.epsilon *= self.epsilon_decay
            
    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        # # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        # batch = random.sample(self.memory, self.batch_size)
        # states, actions, rewards, next_states, dones = zip(*batch)
      
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)

            
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        #states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        #next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        #actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        #rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        #dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        #q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        #  1. Double DQN loss function
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1)
            
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # Q_target(s, a) = r + gamma * Q_target(s', argmax_a' Q(s', a'))
            target_q_values = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q_values
            
        # loss = nn.MSELoss()(q_values, target_q_values)

        td_errors = target_q_values - q_values
        weights = weights.to(self.device)

        loss = (weights * td_errors ** 2).mean()


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory.update_priorities(indices, td_errors.abs().detach().cpu().numpy())

      
        ########## END OF YOUR CODE ##########  

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
            wandb.log({
                "Train Loss": loss.item(),
                "Q Mean": q_values.mean().item(),
                "Q Std": q_values.std().item()
            }, step=self.env_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=50000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)#0.995 origin:0.999999
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--n_step", type=int, default=2)
    parser.add_argument("--reward_scale", type=int, default=1)

    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args=args)
    agent.run(args.episodes)
    
    
''' command to run the script
python dqn.py --wandb-run-name debug-fix               --replay-start-size 1000               --epsilon-decay 0.995               --max-episode-steps 500               --train-per-step 1
'''