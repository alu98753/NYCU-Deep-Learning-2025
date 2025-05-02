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

# C51 support (51 atoms)
N_ATOMS = 51
V_MIN, V_MAX = -10.0, 10.0
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)
# Ensure SUPPORT is on the correct device later in the agent
# SUPPORT = torch.linspace(V_MIN, V_MAX, N_ATOMS)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# <<<--- 原來的 DQN Class (可以保留或刪除) --->>>
# class DQN(nn.Module):
#     def __init__(self, num_actions):
#         super(DQN, self).__init__()
#         self.num_actions = num_actions
#         self.base = nn.Sequential(
#             nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
#             nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
#             nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
#             nn.Flatten(), nn.Linear(64*7*7, 512), nn.ReLU(),
#         )
#         self.head = nn.Linear(512, num_actions * N_ATOMS)

#     def forward(self, x):
#         x = x / 255.0
#         feat = self.base(x)
#         logits = self.head(feat)  # [B, num_actions * N_ATOMS]
#         return logits.view(-1, self.num_actions, N_ATOMS)  # [B, A, N_ATOMS]


# <<<--- 新的 Dueling C51 DQN Class --->>>
class DuelingC51DQN(nn.Module):
    def __init__(self, num_actions):
        super(DuelingC51DQN, self).__init__()
        self.num_actions = num_actions
        self.n_atoms = N_ATOMS

        # Shared feature extraction base
        self.base = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), # Output shape: [B, 512]
            nn.ReLU()
        )

        # Value stream head
        self.value_head = nn.Linear(512, self.n_atoms) # Output shape: [B, N_ATOMS]

        # Advantage stream head
        self.advantage_head = nn.Linear(512, num_actions * self.n_atoms) # Output shape: [B, A * N_ATOMS]

    def forward(self, x):
        # Normalize input
        x = x / 255.0

        # Get shared features
        feat = self.base(x) # Shape: [B, 512]

        # Calculate value logits
        value_logits = self.value_head(feat) # Shape: [B, N_ATOMS]
        # Reshape for broadcasting: [B, 1, N_ATOMS]
        value_logits = value_logits.unsqueeze(1)

        # Calculate advantage logits
        advantage_logits = self.advantage_head(feat) # Shape: [B, A * N_ATOMS]
        # Reshape to [B, A, N_ATOMS]
        advantage_logits = advantage_logits.view(-1, self.num_actions, self.n_atoms)

        # Calculate mean advantage logits across actions
        # Shape: [B, 1, N_ATOMS]
        mean_advantage_logits = advantage_logits.mean(1, keepdim=True)

        # Combine value and advantage streams (Dueling formula applied to logits)
        # Q_logits(s, a) = V_logits(s) + (A_logits(s, a) - mean(A_logits(s, a')))
        # Broadcasting rules:
        # value_logits:       [B, 1, N_ATOMS]
        # advantage_logits:   [B, A, N_ATOMS]
        # mean_advantage:     [B, 1, N_ATOMS]
        q_logits = value_logits + (advantage_logits - mean_advantage_logits)
        # Final shape: [B, A, N_ATOMS]

        return q_logits # Return the logits for the distribution of each action's Q-value


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


### Sum Tree
import numpy

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

        # Check if dataIdx is valid before accessing self.data
        if dataIdx < 0 or dataIdx >= self.capacity:
             # This might happen if tree structure is inconsistent or during edge cases
             # Handle appropriately, e.g., return a default or re-sample
             # For now, let's return an indicator of failure or the root node info
             print(f"Warning: Invalid data index {dataIdx} retrieved for s={s}. Tree index: {idx}")
             # Returning None or raising an error might be better depending on how .sample handles it
             # For simplicity, let's return something that .sample's while loop will catch
             return (idx, self.tree[idx], 0) # Return 0 as placeholder data


        # Check if the retrieved data is actually a transition tuple
        # self.data[dataIdx] might still be the initial numpy.zeros object if not filled
        retrieved_data = self.data[dataIdx]
        # if isinstance(retrieved_data, (int, float)) and retrieved_data == 0:
        # A more robust check might be needed if 0 is a valid value in your state/action etc.
        # Check if it's the default object type:
        if isinstance(retrieved_data, np.ndarray) and retrieved_data.dtype == object and retrieved_data == 0:
             # print(f"Warning: Retrieved default data object at index {dataIdx}. Re-sampling might be needed.")
             # Let the caller handle this, maybe by checking the type.
             pass # Or return a specific marker

        return (idx, self.tree[idx], retrieved_data)

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
        self.epsilon = 1e-6 # Small constant added to priorities
        self.reward_scale = reward_scale # Used for scaling error before calculating priority
        self.beta_increment_per_sampling = 0.005
        self.max_priority = 1.0 # Initial max priority

    def __len__(self):
        return self.tree.n_entries

    def add(self, transition, error=None):
        # If error is None, use max priority for new samples
        # This ensures new samples have a high chance of being selected initially
        priority = self.max_priority if error is None else (abs(error / self.reward_scale) + self.epsilon) ** self.alpha
        self.tree.add(priority, transition)


    def sample(self, batch_size):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total() / batch_size
        self.beta = min(1., self.beta + self.beta_increment_per_sampling) # Anneal beta

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, priority, data = self.tree.get(s)

            # Handle cases where get might return invalid data (e.g., initial zeros)
            # Keep resampling until valid data is found
            # Check if data is a tuple (expected transition format)
            retry_count = 0
            max_retries = 10 # Avoid infinite loops
            while not isinstance(data, tuple) and retry_count < max_retries:
                 # print(f"Warning: Resampling needed. Got data: {data} (type: {type(data)}) for s={s}")
                 new_s = random.uniform(0, self.tree.total()) # Resample from the entire range
                 idx, priority, data = self.tree.get(new_s)
                 retry_count += 1
            if not isinstance(data, tuple):
                 # If still not valid after retries, something might be wrong
                 # Or the buffer might be sparsely populated
                 print(f"Error: Failed to sample valid data after {max_retries} retries.")
                 # Handle error: maybe return None or raise exception
                 # For now, try to continue, but this might lead to issues downstream
                 # As a fallback, sample completely randomly from filled entries if possible
                 if self.tree.n_entries > 0:
                     random_data_idx = random.randrange(self.tree.n_entries)
                     data = self.tree.data[random_data_idx]
                     # We don't have the correct tree index or priority here easily
                     # This part needs careful handling. Let's skip this sample for now.
                     continue # Skip to next sample in the batch

            if data is not None and isinstance(data, tuple): # Ensure data is valid
                priorities.append(priority)
                batch.append(data)
                idxs.append(idx)
            #else:
                # Handle the case where data is None or invalid after retries
                # print(f"Skipping sample due to invalid data.")
                # Need to decide how to handle incomplete batches

        if not batch: # If the batch is empty after sampling attempts
             print("Warning: Sampled empty batch.")
             return None, None, None, None, None, None, None # Or handle appropriately

        sampling_probabilities = np.array(priorities) / self.tree.total()

        # Importance Sampling (IS) weights
        # w_i = (N * P(i)) ^ (-beta) / max_j(w_j)
        weights = (len(self) * sampling_probabilities) ** (-self.beta)
        weights /= weights.max() # Normalize by max weight for stability

        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        return (states, torch.from_numpy(actions).long(),
                torch.from_numpy(rewards).float(),
                next_states, torch.from_numpy(dones).float(),
                torch.from_numpy(weights).float(), idxs)


    def update_priorities(self, indices, errors):
        priorities = (np.abs(errors / self.reward_scale) + self.epsilon) ** self.alpha
        # priorities = np.clip(priorities, 1e-6, 100) # Optional: Clip priorities

        for idx, priority in zip(indices, priorities):
            # Ensure priority is a float, not an array element if errors was numpy array
            p_val = float(priority)
            self.tree.update(idx, p_val)
            # Update max priority observed
            self.max_priority = max(self.max_priority, p_val)

# --- Rest of the DQNAgent class ---
class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # <<<--- Use the new DuelingC51DQN class --->>>
        self.q_net = DuelingC51DQN(self.num_actions).to(self.device)
        self.q_net.apply(init_weights) # Apply weight initialization
        self.target_net = DuelingC51DQN(self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval() # Target network in evaluation mode

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4) # Use AdamW or add weight decay if needed

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.epsilon_start = args.epsilon_start
        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21 # Initialize best reward correctly
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = os.path.join(
            args.save_dir,
            f"{args.wandb_run_name}_{time.strftime('%Y%m%d-%H%M%S')}",
            f"{env_name.replace('/', '_')}" # Ensure env_name is filesystem friendly
        )
        os.makedirs(self.save_dir, exist_ok=True)

        self.memory = PrioritizedReplayBuffer(args.memory_size, alpha=args.per_alpha, beta=args.per_beta, reward_scale=args.reward_scale)

        self.n_step = getattr(args, "n_step", 3) # Default n-step=3 is common
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.reward_scale = args.reward_scale

        # Move SUPPORT tensor to the correct device
        self.support = torch.linspace(V_MIN, V_MAX, N_ATOMS).to(self.device)
        self.delta_z = DELTA_Z # Store delta_z
        self.frame_skip = 6 # Frame skip for Atari games

    def select_action(self, state, evaluation=False):
        # Epsilon-greedy for exploration during training
        if not evaluation and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        # Greedy action selection for evaluation or exploitation
        s = torch.from_numpy(state).unsqueeze(0).to(self.device).float() # Ensure float type
        with torch.no_grad():
            q_logits = self.q_net(s)                     # [1, A, N_ATOMS]
            probabilities = torch.softmax(q_logits, dim=2) # [1, A, N_ATOMS]
            # Calculate expected Q-value for each action
            q_values = (probabilities * self.support).sum(2) # [1, A]
            # Select action with the highest expected Q-value
            return q_values.argmax(1).item()

    def env_step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = {}
        for _ in range(self.frame_skip):
            next_obs, reward, terminated, trunc, info = self.env.step(action)
            total_reward += reward
            done = terminated or trunc
            if done:
                break
        return next_obs, total_reward, terminated, trunc, info


    def test_env_step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        info = {}
        for _ in range(self.frame_skip):
            next_obs, reward, terminated, trunc, info = self.test_env.step(action)
            total_reward += reward
            done = terminated or trunc
            if done:
                break
        return next_obs, total_reward, terminated, trunc, info

    def _compute_initial_td_error(self, state, action, reward, next_state, done):
        """Computes TD error for a single transition to initialize priority."""
        # This is approximate, ideally use the full C51 calculation,
        # but a simple Q-value difference can work for initial priority.
        with torch.no_grad():
            s_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            s_next_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)

            # Current Q-dist estimation
            q_logits = self.q_net(s_tensor) # [1, A, N]
            probs = torch.softmax(q_logits, dim=2)
            q_values = (probs * self.support).sum(2) # [1, A]
            q_val_current = q_values[0, action]

            # Target Q-dist estimation (using Double DQN logic)
            q_logits_next_online = self.q_net(s_next_tensor) # [1, A, N]
            probs_next_online = torch.softmax(q_logits_next_online, dim=2)
            q_values_next_online = (probs_next_online * self.support).sum(2) # [1, A]
            next_action = q_values_next_online.argmax(1) # [1]

            q_logits_next_target = self.target_net(s_next_tensor) # [1, A, N]
            probs_next_target = torch.softmax(q_logits_next_target, dim=2)
            # Use target network's distribution for the action selected by the online network
            prob_dist_next = probs_next_target[0, next_action.item()] # [N]

            # Project the target distribution
            Tz = reward + (1 - done) * (self.gamma ** self.n_step) * self.support
            Tz = Tz.clamp(V_MIN, V_MAX)
            b = (Tz - V_MIN) / self.delta_z
            l = b.floor().long().clamp(0, N_ATOMS - 1)
            u = b.ceil().long().clamp(0, N_ATOMS - 1)

            m = torch.zeros_like(prob_dist_next) # Shape: [N]
            # Simplified projection loop (no batch dim)
            for i in range(N_ATOMS):
                 m[l[i]] += prob_dist_next[i] * (u[i].float() - b[i])
                 m[u[i]] += prob_dist_next[i] * (b[i] - l[i].float())

            # Calculate expected value of the projected distribution
            target_q_val = (m * self.support).sum()

            td_error = abs(q_val_current - target_q_val)

            # Alternatively, calculate KL divergence based TD error (more aligned with C51 loss)
            # log_probs_current = torch.log_softmax(q_logits[0, action], dim=0) # [N]
            # kl_div = -(m * log_probs_current).sum()
            # td_error = kl_div # Use KL divergence as error measure

        return td_error.item()


    def run(self, episodes):
        total_steps = 0
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            truncated = False
            total_reward = 0
            ep_steps = 0
            self.n_step_buffer.clear() # Clear buffer for new episode

            while not done and not truncated:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env_step(action)
                ep_steps += 1
                total_steps += 1
                self.env_count += 1

                # Clip reward and scale (common practice in Atari)
                original_reward = reward
                reward = np.clip(reward, -1, 1) * self.reward_scale

                next_state = self.preprocessor.step(next_obs)
                done = terminated # Use terminated, truncated handled by loop condition

                # Store experience for n-step calculation
                self.n_step_buffer.append((state, action, reward, next_state, done))

                # If buffer has n steps, calculate return and add to memory
                if len(self.n_step_buffer) == self.n_step:
                    R, S, A = 0, self.n_step_buffer[0][0], self.n_step_buffer[0][1]
                    is_done_n_step = False
                    # Calculate n-step discounted return
                    for i in range(self.n_step):
                        r_i = self.n_step_buffer[i][2]
                        d_i = self.n_step_buffer[i][4]
                        R += (self.gamma ** i) * r_i
                        if d_i:
                            is_done_n_step = True
                            break # Stop accumulating reward if episode terminates within n steps
                    # The next state S_next is from the last element in the buffer
                    S_next = self.n_step_buffer[-1][3]

                    # Compute initial TD error for PER priority
                    # We use the n-step transition (S, A, R_n_step, S_next, is_done_n_step)
                    initial_td_error = self._compute_initial_td_error(S, A, R, S_next, float(is_done_n_step))

                    # Add the n-step transition to the prioritized replay buffer
                    self.memory.add((S, A, R, S_next, float(is_done_n_step)), initial_td_error)

                # Set current state for next iteration
                state = next_state

                # Training step(s)
                if total_steps >= self.replay_start_size:
                    for _ in range(self.train_per_step):
                        loss, td_errors_batch = self.train()
                        # Log training loss etc. if train() returns them
                        if loss is not None and self.train_count % 100 == 0: # Log loss periodically
                            wandb.log({
                                "Train Step": self.train_count,
                                "Training Loss": loss,
                                "Mean TD Error (batch)": np.mean(td_errors_batch) if td_errors_batch is not None else 0,
                                "PER Beta": self.memory.beta,
                            }, step=total_steps)

                if self.env_count % 1000 == 0:        
                    # snapshot frequency: 200k
                    if self.env_count % 200000 == 0 and self.env_count in [200000, 400000, 600000, 800000, 1000000]:
                        model_path = os.path.join(self.save_dir, f"LAB5_313554044_task3_pong{self.env_count}.pt")
                        torch.save(self.q_net.state_dict(), model_path)    

                    print(f"[Collect] Ep: {ep} Step: {ep_steps} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "TD Error": np.mean(td_errors_batch) if td_errors_batch is not None else 0,
                        "Episode": ep,
                        "Step Count": ep_steps,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })


                # Update target network periodically
                if self.train_count > 0 and self.train_count % self.target_update_frequency == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())
                    print(f"--- Updated Target Network at training step {self.train_count} ---")


                # Decay epsilon (linear decay might be better than exponential for long training)
                # Example Linear Decay:
                if total_steps > self.replay_start_size:
                    if self.epsilon > self.epsilon_min:
                        self.epsilon *= self.epsilon_decay
                # Exponential Decay (as originally used):
                # if total_steps > self.replay_start_size and self.epsilon > self.epsilon_min:
                #      self.epsilon *= self.epsilon_decay


                # Handle episode truncation (max steps per episode)
                if ep_steps >= self.max_episode_steps:
                    truncated = True

                # Accumulate original reward for episode logging
                total_reward += original_reward


            # --- End of Episode ---

            # Flush remaining transitions from n-step buffer at the end of the episode
            while len(self.n_step_buffer) > 0:
                R, S, A = 0, self.n_step_buffer[0][0], self.n_step_buffer[0][1]
                is_done_n_step = False
                current_n = len(self.n_step_buffer) # Actual number of steps left
                for i in range(current_n):
                    r_i = self.n_step_buffer[i][2]
                    d_i = self.n_step_buffer[i][4]
                    R += (self.gamma ** i) * r_i
                    if d_i:
                        is_done_n_step = True
                        break
                S_next = self.n_step_buffer[-1][3] # Last state is the next state

                # Compute initial TD error for this partial n-step transition
                # Note: gamma exponent should reflect actual_n if used in error calc
                initial_td_error = self._compute_initial_td_error(S, A, R, S_next, float(is_done_n_step))

                self.memory.add((S, A, R, S_next, float(is_done_n_step)), initial_td_error)
                self.n_step_buffer.popleft() # Remove the processed step


            print(f"[Episode End] Ep: {ep}, Steps: {ep_steps}, Total Reward: {total_reward:.2f}, Total Steps: {total_steps}, Epsilon: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Episode Reward": total_reward,
                "Episode Steps": ep_steps,
                "Total Steps": total_steps,
                "Epsilon": self.epsilon,
                # "Consecutive Hits": self.consecutive_hits # If using custom reward shaping
            }, step=total_steps) # Log against total steps for consistent x-axis


            # Evaluate periodically (e.g., every 5 episodes)
            if ep % 5 == 0:
                eval_reward = self.evaluate()
                print(f"[Evaluation] Ep: {ep}, Avg Reward: {eval_reward:.2f}")
                wandb.log({
                    "Eval Reward": eval_reward,
                    "Episode": ep,
                }, step=total_steps)

                # Save best model based on evaluation reward
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
    def evaluate(self, num_episodes=30):
        """Evaluate the agent's performance."""
        total_rewards = []
        for ep in range(num_episodes):
            obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(obs)
            done, truncated = False, False
            episode_reward = 0
            ep_steps = 0
            while not done and not truncated:
                action = self.select_action(state, evaluation=True) # Use greedy policy
                next_obs, reward, terminated, truncated, _ = self.test_env_step(action)
                done = terminated
                episode_reward += reward
                state = self.preprocessor.step(next_obs)
                ep_steps +=1
                if ep_steps >= self.max_episode_steps: # Add truncation limit here too
                     truncated = True
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)


    def train(self):
        # Check if buffer has enough samples and if it's time to train
        if len(self.memory) < self.replay_start_size:
            return None, None # Return None if not ready to train

        self.train_count += 1

        # Sample a batch from PER buffer
        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)

        # Check if sampling failed (e.g., empty batch)
        if states is None:
             print("Skipping training step due to sampling failure.")
             return None, None

        # Convert to tensors and move to device
        states = torch.from_numpy(states).to(self.device).float()
        next_states = torch.from_numpy(next_states).to(self.device).float()
        actions = actions.to(self.device) # Shape: [B]
        rewards = rewards.to(self.device).unsqueeze(1) # Shape: [B, 1]
        dones = dones.to(self.device).unsqueeze(1) # Shape: [B, 1]
        weights = weights.to(self.device).unsqueeze(1) # Shape: [B, 1]

        # --- Compute Target Distribution (m) ---
        with torch.no_grad():
            # 1. Select best actions in next_state using ONLINE network (Double DQN)
            q_logits_next_online = self.q_net(next_states)           # [B, A, N]
            probs_next_online = torch.softmax(q_logits_next_online, dim=2)
            q_values_next_online = (probs_next_online * self.support).sum(2) # [B, A]
            next_actions = q_values_next_online.argmax(1)          # [B]

            # 2. Get next state value distribution from TARGET network for selected actions
            q_logits_next_target = self.target_net(next_states)      # [B, A, N]
            probs_next_target = torch.softmax(q_logits_next_target, dim=2)
            # Gather the distributions corresponding to the best actions chosen by online net
            # probs_next_target shape [B, N], requires indexing correctly
            prob_dist_next = probs_next_target[torch.arange(self.batch_size), next_actions] # [B, N]

            # 3. Project the target distribution onto the support Z
            # Calculate the projected support Tz = R + gamma^n * Z for non-terminal states
            Tz = rewards + (1 - dones) * (self.gamma ** self.n_step) * self.support.unsqueeze(0) # [B, N]
            Tz = Tz.clamp(V_MIN, V_MAX) # Clamp to [V_min, V_max]

            # Calculate indices and weights for projection
            b = (Tz - V_MIN) / self.delta_z # Shape: [B, N]
            l = b.floor().long()           # Shape: [B, N]
            u = b.ceil().long()            # Shape: [B, N]

            # Correct clamping needed if b is exactly an integer
            # l == u cases need careful handling, but usually small probability
            l = l.clamp(0, N_ATOMS - 1)
            u = u.clamp(0, N_ATOMS - 1)

            # Calculate projection weights (bilinear interpolation)
            # Weight for lower bin (l): prob * (u - b)
            # Weight for upper bin (u): prob * (b - l)
            m = torch.zeros_like(prob_dist_next) # Target distribution, shape: [B, N]

            # Efficient batch projection using index_add_
            # Create offset for batch indexing
            offset = torch.linspace(0, (self.batch_size - 1) * N_ATOMS, self.batch_size).long().unsqueeze(1).to(self.device) # Shape: [B, 1]

            # Add weights to lower bins (l)
            m.view(-1).index_add_(0, (l + offset).view(-1), (prob_dist_next * (u.float() - b)).view(-1))
            # Add weights to upper bins (u)
            m.view(-1).index_add_(0, (u + offset).view(-1), (prob_dist_next * (b - l.float())).view(-1))
            # m now contains the projected target distributions for the batch

        # --- Compute Loss ---
        # Get current Q-distribution logits from ONLINE network for the taken actions
        q_logits_current = self.q_net(states) # [B, A, N]
        # Gather the logits for the specific actions taken
        log_probs_current = torch.log_softmax(q_logits_current, dim=2) # Use log_softmax for numerical stability
        log_p_a = log_probs_current[torch.arange(self.batch_size), actions] # [B, N]

        # Calculate Cross-Entropy Loss between target distribution (m) and current (log_p_a)
        # Loss = - sum(m * log_p_a)
        loss_individual = -(m * log_p_a).sum(1) # Shape: [B]

        # Apply Importance Sampling weights (from PER)
        loss = (weights.squeeze(1) * loss_individual).mean()

        # --- Optimization Step ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0) # Gradient clipping
        self.optimizer.step()

        # --- Update Priorities in PER Buffer ---
        # Calculate TD errors (e.g., using KL divergence or absolute difference for priorities)
        # Using the cross-entropy loss itself (before weighting) as the error measure is common
        td_errors_numpy = loss_individual.detach().cpu().numpy()

        # Update priorities in the buffer
        self.memory.update_priorities(indices, td_errors_numpy)

        return loss.item(), td_errors_numpy # Return loss and errors for logging



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5", help="Gym environment name")
    parser.add_argument("--save-dir", type=str, default="./results", help="Directory to save models and logs")
    parser.add_argument("--wandb-run-name", type=str, default="pong_dueling_c51_per_ddqn_nstep", help="W&B run name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--memory-size", type=int, default=100000, help="Replay buffer size") # Often larger for Atari
    parser.add_argument("--lr", type=float, default=0.0000625, help="Learning rate (Rainbow uses 6.25e-5)") # Adjusted LR
    parser.add_argument("--discount-factor", type=float, default=0.99, help="Gamma (discount factor)")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon for exploration")
    # Epsilon decay parameters - consider linear decay over 1M steps
    parser.add_argument("--epsilon-decay", type=float, default=0.9999, help="Exponential decay rate (if used)")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum epsilon value") # Lower min epsilon often used
    parser.add_argument("--target-update-frequency", type=int, default=8000, help="Frequency (in training steps) to update target network") # Slower updates often better
    parser.add_argument("--replay-start-size", type=int, default=20000, help="Number of steps to fill buffer before training starts") # Larger start size
    parser.add_argument("--max-episode-steps", type=int, default=10000, help="Max steps per episode (can prevent infinite loops)") # Or use env's default
    parser.add_argument("--train-per-step", type=int, default=1, help="Number of training updates per environment step (often 1)") # Can be fractional e.g. update every 4 steps
    parser.add_argument("--episodes", type=int, default=2000, help="Total number of episodes to train for") # Train longer
    parser.add_argument("--n_step", type=int, default=3, help="N-step return horizon")
    parser.add_argument("--reward_scale", type=int, default=1, help="Scale factor for rewards (often 1 after clipping)")
    # PER parameters
    parser.add_argument("--per-alpha", type=float, default=0.5, help="Alpha for PER priority calculation") # Rainbow uses 0.5
    parser.add_argument("--per-beta", type=float, default=0.4, help="Initial beta for PER importance sampling") # Rainbow uses 0.4, anneals to 1.0

    args = parser.parse_args()

    # Update run name to include Dueling
    if "dueling" not in args.wandb_run_name.lower():
         args.wandb_run_name += "_dueling"

    wandb.init(config=args, project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)

    # Log the final configuration used
    config = wandb.config
    print("Running with configuration:")
    print(config)

    # Pass the specific env_name from args
    agent = DQNAgent(env_name=args.env_name, args=args)

    agent.run(args.episodes)

    wandb.finish()
    print("Training finished.")

''' Example Command:
python dqn_task3_distribution_duel.py --env-name "ALE/Pong-v5" \
                           --wandb-run-name "task3-distribution-duel" \
                           --replay-start-size 1000 \
                           --target-update-frequency 1000 \
                           --lr 0.0001 \
                           --epsilon-min 0.05 \
                           --per-alpha 0.6 \
                           --per-beta 0.4 \
                           --n_step 3 \
                           --episodes 2000
'''