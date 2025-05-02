# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL - C51 DQN Adaptation for Fast & Stable Learning
# Contributors: Wei Hung and Alison Wen (Modified based on Gemini feedback)
# Instructor: Ping-Chun Hsieh

from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy # Use numpy directly
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time
import math
from typing import Deque, Dict, List, Tuple
import gc
import pickle # For loading expert data

gym.register_envs(ale_py)

# --- Limit Threading (Keep from previous version) ---
NUM_THREADS = "4"
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
torch.set_num_threads(int(NUM_THREADS))
torch.set_num_interop_threads(int(NUM_THREADS))
cv2.setNumThreads(0)
# --------------------

# C51 support (Global constants for clarity)
N_ATOMS = 51
# Adjusted V_MIN/V_MAX slightly based on potential N-step returns range with clipping
# Needs verification based on actual returns observed if issues persist
V_MIN = -5.0
V_MAX = 5.0
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)
# Define SUPPORT on the default device first, move later in Agent
SUPPORT_NP = numpy.linspace(V_MIN, V_MAX, N_ATOMS)


def init_weights(m):
    """Initialize weights using Kaiming uniform for Conv/Linear layers."""
    # Assuming NoisyLinear is not used in this C51 version
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# --- C51 DQN Network ---
class DQN(nn.Module):
    """ C51 DQN Network """
    def __init__(self, num_actions, frame_stack=4): # Added frame_stack for consistency
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.frame_stack = frame_stack # Store frame_stack
        self.base = nn.Sequential(
            nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512), nn.ReLU(),
        )
        # Output layer for C51: produces logits for each atom for each action
        self.head = nn.Linear(512, num_actions * N_ATOMS)

    def forward(self, x):
        # Normalize input images
        x = x / 255.0
        feat = self.base(x)
        logits = self.head(feat)
        # Reshape to (batch_size, num_actions, num_atoms)
        return logits.view(-1, self.num_actions, N_ATOMS)

    def get_q_values(self, x):
        """ Helper to get expected Q-values from C51 output """
        logits = self(x)
        probabilities = torch.softmax(logits, dim=2)
        # Ensure SUPPORT is on the same device
        support = torch.tensor(SUPPORT_NP, device=x.device, dtype=torch.float32)
        q_values = (probabilities * support).sum(2) # Expected Q-value
        return q_values

# --- Atari Preprocessor (Keep from previous version) ---
class AtariPreprocessor:
    """ Preprocessing the state input for Atari """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        """ Convert to grayscale and resize """
        if len(obs.shape) == 1:
             return obs # Should not happen for Atari
        if obs.dtype != numpy.uint8:
             obs = obs.astype(numpy.uint8)
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized # Returns shape (84, 84)

    def reset(self, obs):
        """ Reset frame buffer with the first observation """
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return numpy.stack(self.frames, axis=0) # Returns shape (frame_stack, 84, 84)

    def step(self, obs):
        """ Process a new observation and add to frame buffer """
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return numpy.stack(self.frames, axis=0) # Returns shape (frame_stack, 84, 84)


# --- SumTree (Refactored for Priorities Only - from previous step) ---
class SumTree:
    write = 0
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.n_entries = 0
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0: self._propagate(parent, change)
    def _retrieve(self, idx, s):
        left, right = 2 * idx + 1, 2 * idx + 2
        if left >= len(self.tree): return idx
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(right, s - self.tree[left])
    def total(self): return self.tree[0]
    def add(self, p):
        idx = self.write + self.capacity - 1
        data_idx = self.write
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity: self.n_entries += 1
        return data_idx
    def update(self, idx, p):
        if not (self.capacity - 1 <= idx < 2 * self.capacity - 1):
             print(f"Warning: Attempting SumTree.update on non-leaf index {idx}")
             return # Avoid updating non-leaf directly
        p = max(p, 1e-6) # Ensure positive
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        if not (0 <= dataIdx < self.capacity):
             print(f"Error: SumTree.get invalid data index {dataIdx}. Clamping.")
             dataIdx = max(0, min(dataIdx, self.capacity - 1))
        return (idx, self.tree[idx], dataIdx)

# --- PrioritizedReplayBuffer (Refactored with Typed Arrays - from previous step) ---
class PrioritizedReplayBuffer:
    """ PER using SumTree for priorities and separate typed arrays for data. """
    def __init__(self, capacity, frame_stack, alpha=0.6, beta_start=0.4, total_train_steps=1000000, reward_scale=1.0, epsilon=1e-6):
        print(f"Initializing PrioritizedReplayBuffer: capacity={capacity}, frame_stack={frame_stack}")
        if reward_scale <= 0:
             print(f"Warning: reward_scale must be positive. Setting to 1.0.")
             reward_scale = 1.0
        self.capacity = capacity
        self.frame_stack = frame_stack
        self.alpha = alpha
        self.beta = beta_start
        self.beta_final = 1.0
        self.epsilon = epsilon
        self.reward_scale = reward_scale

        # Typed arrays
        self.states = numpy.empty((capacity, frame_stack, 84, 84), dtype=numpy.uint8)
        self.next_states = numpy.empty((capacity, frame_stack, 84, 84), dtype=numpy.uint8)
        self.actions = numpy.empty(capacity, dtype=numpy.int64)
        self.rewards = numpy.empty(capacity, dtype=numpy.float32)
        self.dones = numpy.empty(capacity, dtype=numpy.bool_)

        self.tree = SumTree(capacity)

        # Beta annealing (based on total *env* steps for consistency)
        self.total_env_steps_for_beta = total_train_steps # Use env steps from args
        if self.total_env_steps_for_beta > 0:
             # Calculate increment per *environment* step for smoother annealing
             self.beta_increment_per_env_step = (self.beta_final - self.beta) / self.total_env_steps_for_beta
        else:
             self.beta_increment_per_env_step = 0
        print(f"PER Beta annealing: start={self.beta}, final={self.beta_final}, increment per ENV step={self.beta_increment_per_env_step:.8e} over {self.total_env_steps_for_beta} steps")

    def __len__(self):
        return self.tree.n_entries

    def update_beta(self):
        """ Update beta based on environment steps. Call this periodically in the agent's run loop. """
        if self.beta < self.beta_final:
             self.beta = min(self.beta_final, self.beta + self.beta_increment_per_env_step)

    def add(self, transition, error):
        """ Adds transition with priority derived from error. """
        state, action, reward, next_state, done = transition
        priority = (abs(error / self.reward_scale) + self.epsilon) ** self.alpha
        priority = max(priority, self.epsilon)
        data_idx = self.tree.add(priority)
        try:
            self.states[data_idx] = state
            self.actions[data_idx] = action
            self.rewards[data_idx] = reward
            self.next_states[data_idx] = next_state
            self.dones[data_idx] = done
        except IndexError: print(f"Error: Index {data_idx} out of bounds during add.")

    def sample(self, batch_size):
        """ Samples batch, returns data and SumTree indices. """
        if self.tree.n_entries < batch_size: # Ensure enough entries to sample
             print(f"Warning: Not enough entries ({self.tree.n_entries}) in buffer to sample batch size {batch_size}.")
             return None

        current_size = self.tree.n_entries
        batch_indices = numpy.empty(batch_size, dtype=numpy.int32)
        tree_indices = numpy.empty(batch_size, dtype=numpy.int32)
        priorities = numpy.empty(batch_size, dtype=numpy.float32)
        segment = self.tree.total() / batch_size

        samples_collected = 0
        max_loop_attempts = batch_size * 5 # Prevent infinite loop
        current_loop_attempt = 0

        while samples_collected < batch_size and current_loop_attempt < max_loop_attempts:
            current_loop_attempt +=1
            i = samples_collected # Index for this sample
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, min(b, self.tree.total()))
            s = min(s, self.tree.total())

            try:
                tree_idx, priority, data_idx = self.tree.get(s)
                if data_idx >= current_size: # Should not happen if SumTree is correct
                    # print(f"Warning: Sampled invalid data index {data_idx} >= size {current_size}. Retrying...")
                    continue # Retry sampling for this slot

                # Check for duplicates - simple check, more robust needed for large batches/many retries
                # if data_idx in batch_indices[:samples_collected]:
                #     print(f"Warning: Duplicate index {data_idx} sampled. Retrying...")
                #     continue

                batch_indices[samples_collected] = data_idx
                tree_indices[samples_collected] = tree_idx
                priorities[samples_collected] = priority
                samples_collected += 1

            except Exception as e:
                print(f"Error during sample retrieval: {e}. Skipping sample {i}.")
                # Continue trying to fill the batch if possible
                continue

        if samples_collected < batch_size:
            print(f"Error: Failed to collect enough samples ({samples_collected}/{batch_size}) after {max_loop_attempts} attempts.")
            return None

        # IS weights
        sampling_probabilities = priorities / self.tree.total()
        weights = numpy.power(current_size * sampling_probabilities, -self.beta)
        if weights.max() > 1e-9: weights /= weights.max()
        else: weights = numpy.ones_like(weights)

        # Retrieve data
        batch_states = self.states[batch_indices]
        batch_actions = self.actions[batch_indices]
        batch_rewards = self.rewards[batch_indices]
        batch_next_states = self.next_states[batch_indices]
        batch_dones = self.dones[batch_indices]

        # Convert to tensors (states remain numpy)
        actions_t = torch.from_numpy(batch_actions)
        rewards_t = torch.from_numpy(batch_rewards)
        dones_t = torch.from_numpy(batch_dones).to(dtype=torch.float32)
        weights_t = torch.from_numpy(weights).to(dtype=torch.float32)

        return batch_states, actions_t, rewards_t, batch_next_states, dones_t, weights_t, tree_indices

    def update_priorities(self, indices, errors):
        """ Updates priorities for given SumTree indices. """
        if isinstance(errors, torch.Tensor): errors = errors.abs().detach().cpu().numpy()
        else: errors = numpy.abs(numpy.array(errors))
        for i, idx in enumerate(indices):
            priority = (abs(errors[i] / self.reward_scale) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)


# --- DQNAgent (Modified for C51, Soft Updates, Stability) ---
class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        if args is None: raise ValueError("Agent requires arguments (args).")
        self.args = args
        self.env_name = env_name
        self.seed = args.seed

        # Device and Env Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # Use the make_env helper to ensure consistent settings
        self.env = make_env(env_name, seed=self.seed)
        self.test_env = make_env(env_name, seed=self.seed + 1) # Use different seed for test env
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor(frame_stack=args.frame_stack)

        # Network Setup (C51)
        self.q_net = DQN(self.num_actions, frame_stack=args.frame_stack).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions, frame_stack=args.frame_stack).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval() # Target network is only for inference

        # Optimizer (Adam with potentially higher eps)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=args.adam_eps)

        # Hyperparameters from args
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_final = args.epsilon_min # Rename arg for clarity
        self.epsilon_decay_rate = args.epsilon_decay_rate # Calculated in main
        self.replay_start_size = args.replay_start_size
        self.train_per_step = args.train_per_step
        self.max_episode_steps = args.max_episode_steps

        # Soft Target Update Tau (scaled as per guide, inverse relation)
        # If train_per_step is 1, tau is args.soft_tau. If 4, tau is args.soft_tau / 4.
        self.soft_tau = args.soft_tau 
        print(f"Using Soft Target Updates with tau = {self.soft_tau:.5f} (base={args.soft_tau}, train_per_step={self.train_per_step})")

        # Replay Buffer (Refactored PER)
        self.memory = PrioritizedReplayBuffer(
            capacity=args.memory_size,
            frame_stack=args.frame_stack,
            alpha=args.alpha,
            beta_start=args.beta,
            total_train_steps=args.total_train_steps, # Used for beta annealing based on env steps
            reward_scale=args.reward_scale,
            epsilon=1e-6
        )
        print(f"Initialized PER buffer with capacity {args.memory_size}.")
        self.added_expert_count = 0 # Track added expert transitions

        # Expert Data Loading
        if args.load_expert_data:
            self._load_expert_data(args.load_expert_data)

        # N-step Buffer
        self.n_step = args.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Counters and Tracking
        self.env_count = 0
        self.train_count = 0
        self.best_reward = -float('inf') # More general init
        self.save_dir = os.path.join(args.save_dir, args.wandb_run_name if args.wandb_run_name else f"c51_{time.strftime('%Y%m%d-%H%M%S')}")
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Results and models will be saved in: {self.save_dir}")

        # Move SUPPORT to device
        self.support = torch.tensor(SUPPORT_NP, device=self.device, dtype=torch.float32)


    def _load_expert_data(self, expert_data_path):
        """ Loads expert data from a pickle file and adds it to the buffer. """
        print(f"\n--- Attempting to load expert data from: {expert_data_path} ---")
        try:
            with open(expert_data_path, 'rb') as f:
                expert_experiences = pickle.load(f)
            if not isinstance(expert_experiences, list):
                print("Warning: Loaded expert data not a list. Skipping.")
                return
            print(f"Successfully loaded {len(expert_experiences)} expert transitions.")
        except FileNotFoundError:
            print(f"Error: Expert data file not found: {expert_data_path}. Skipping.")
            return
        except Exception as e:
            print(f"Error loading/unpickling expert data: {e}. Skipping.")
            return

        # Decide how many transitions to add (e.g., up to replay_start_size)
        num_to_add = min(len(expert_experiences), self.args.replay_start_size)
        # Alternative: add a fixed fraction or all available? Let's use replay_start_size.
        print(f"Adding up to {num_to_add} expert transitions to PER...")

        # Add with max priority (use reward_scale as proxy error)
        initial_priority_proxy_error = self.memory.reward_scale
        added_count = 0
        skipped_count = 0
        for transition in expert_experiences[:num_to_add]:
            if isinstance(transition, tuple) and len(transition) == 5:
                # Ensure data format matches (S, A, R, S_next, D)
                # Clip reward if necessary (assuming expert data uses clipped rewards?)
                # If expert reward is raw, clip it here before adding
                s, a, r, s_next, d = transition
                # r_clipped = numpy.clip(r, -1, 1).item() # Clip if needed
                # self.memory.add((s, a, r_clipped, s_next, d), initial_priority_proxy_error)
                self.memory.add(transition, initial_priority_proxy_error) # Assuming expert data reward is already suitable
                added_count += 1
                if added_count % 5000 == 0: print(f"  ... added {added_count}/{num_to_add} expert transitions")
            else:
                skipped_count += 1
                if skipped_count == 1: print("Warning: Skipping invalid transition format in expert data.")

        self.added_expert_count = added_count
        if skipped_count > 0: print(f"Warning: Skipped {skipped_count} invalid transitions.")
        print(f"Finished adding {added_count} expert transitions. Buffer size: {len(self.memory)} / {self.memory.capacity}")
        print("--- Finished expert data loading ---")


    def select_action(self, state):
        """ Select action using epsilon-greedy policy with C51 Q-values """
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net.get_q_values(state_t) # Use helper to get expected Q
            return q_values.argmax(1).item()


    def _calculate_n_step_return(self, current_n_step_buffer):
        """ Calculates N-step return, start state/action, final next state, done flag. """
        R = 0.0
        S, A = current_n_step_buffer[0][:2] # Initial state and action
        D = False
        final_next_state = current_n_step_buffer[-1][3] # The state after n steps
        current_n = len(current_n_step_buffer) # Handle buffers not yet full

        for i in range(current_n):
            s_i, a_i, r_i, s_next_i, done_i = current_n_step_buffer[i]
            R += (self.gamma ** i) * r_i
            if done_i:
                D = True
                # Important: If terminated, the "final_next_state" for Bellman update
                # should be the state *after* the step that terminated.
                final_next_state = s_next_i
                break # Stop accumulating reward

        return S, A, R, final_next_state, D, current_n # Return actual n used


    def run(self, episodes):
        """ Main training loop """
        total_start_time = time.time()
        for ep in range(episodes):
            episode_start_time = time.time()
            obs, _ = self.env.reset(seed=self.seed + ep) # Vary seed per episode
            state = self.preprocessor.reset(obs)
            self.n_step_buffer.clear()
            original_episode_reward = 0
            episode_steps = 0
            done = False

            while not done and episode_steps < self.max_episode_steps:
                # Linear Epsilon Decay based on env_count
                if self.env_count >= self.replay_start_size:
                    self.epsilon = max(self.epsilon_final, self.args.epsilon_start - self.env_count * self.epsilon_decay_rate)
                else:
                    self.epsilon = self.args.epsilon_start # Keep high during initial exploration

                # Select action and step env
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                original_episode_reward += reward # Log unclipped reward

                # Clip reward for storage and learning
                clipped_reward = numpy.clip(reward, -1.0, 1.0).item()

                next_state = self.preprocessor.step(next_obs)
                self.n_step_buffer.append((state, action, clipped_reward, next_state, done))

                # Process N-step buffer when full
                if len(self.n_step_buffer) == self.n_step:
                    S, A, R_n, S_next_n, D_n, actual_n = self._calculate_n_step_return(self.n_step_buffer)
                    # Add to PER buffer with MAX priority (simpler than calculating TD error here)
                    max_error_proxy = self.memory.reward_scale * 2 # Give it high priority
                    self.memory.add((S, A, R_n, S_next_n, D_n), max_error_proxy)

                state = next_state
                self.env_count += 1
                episode_steps += 1
                self.memory.update_beta() # Anneal beta based on env steps

                # --- Training Step ---
                if self.env_count >= self.replay_start_size:
                    for _ in range(self.train_per_step):
                        self.train() # Perform train_per_step updates

                # --- Logging and Saving ---
                log_interval = 1000 # Log progress every 1000 env steps
                if self.env_count % log_interval == 0:
                    log_dict = {
                        "Progress/Env Steps": self.env_count,
                        "Progress/Train Steps": self.train_count,
                        "Parameters/PER Beta": self.memory.beta,
                        "Parameters/Epsilon": self.epsilon,
                    }
                    print(f"[Progress] Env Steps: {self.env_count}, Train Steps: {self.train_count}, "
                          f"Eps: {self.epsilon:.4f}, Beta: {self.memory.beta:.4f}, "
                          f"Buffer: {len(self.memory)}/{self.memory.capacity}")
                    wandb.log(log_dict, step=self.env_count) # Log against env_count

                save_interval = getattr(self.args, "save_interval", 200000)
                if self.env_count > 0 and self.env_count % save_interval == 0:
                    snapshot_path = os.path.join(self.save_dir, f"q_net_snapshot_{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), snapshot_path)
                    print(f"Saved snapshot to {snapshot_path}")

            # --- End of Episode ---
            episode_duration = time.time() - episode_start_time

            # Flush remaining N-step buffer transitions
            while len(self.n_step_buffer) > 0:
                S, A, R_n, S_next_n, D_n, actual_n = self._calculate_n_step_return(self.n_step_buffer)
                max_error_proxy = self.memory.reward_scale * 2
                self.memory.add((S, A, R_n, S_next_n, D_n), max_error_proxy)
                self.n_step_buffer.popleft() # Process from the left

            # Log episode results
            print(f"[Episode End] Ep: {ep+1}/{episodes}, Orig Reward: {original_episode_reward:.2f}, "
                  f"Steps: {episode_steps}, Env Steps: {self.env_count}, Duration: {episode_duration:.2f}s")
            wandb.log({
                "Episode/Episode Number": ep + 1,
                "Reward/Original Episode Reward": original_episode_reward,
                "Episode/Steps": episode_steps,
                "Perf/Episode Duration (s)": episode_duration,
                "Progress/Env Steps": self.env_count, # Log again for alignment
                "Progress/Train Steps": self.train_count,
            }, step=self.env_count)

            # Periodic Evaluation
            eval_freq = getattr(self.args, "eval_frequency_episodes", 10) # Evaluate more often
            if (ep + 1) % eval_freq == 0:
                eval_reward = self.evaluate(num_eval_episodes=getattr(self.args, "eval_episodes", 30)) # Use more eval episodes
                wandb.log({"Reward/Evaluation Reward": eval_reward}, step=self.env_count)
                if eval_reward >= self.best_reward:
                    self.best_reward = eval_reward
                    best_model_path = os.path.join(self.save_dir, "best_q_net.pt")
                    torch.save(self.q_net.state_dict(), best_model_path)
                    print(f"Saved new best model with eval reward {eval_reward:.2f} to {best_model_path}")

        # End of Training Cleanup
        total_duration = time.time() - total_start_time
        print(f"\nTraining finished. Total duration: {total_duration:.2f} seconds")
        self.env.close()
        self.test_env.close()
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


    def evaluate(self, num_eval_episodes=10): # Default to 10 episodes for more stable eval
        """ Evaluate agent's greedy policy. """
        print(f"\nStarting evaluation for {num_eval_episodes} episodes...")
        eval_start_time = time.time()
        episode_rewards = []
        self.q_net.eval() # Set to evaluation mode

        for i in range(num_eval_episodes):
            obs, _ = self.test_env.reset(seed=self.seed + 1000 + i) # Use different seeds
            state = self.preprocessor.reset(obs)
            done = False
            original_episode_reward = 0
            episode_steps = 0
            while not done and episode_steps < self.max_episode_steps:
                # Greedy action selection using expected Q-values from C51
                state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                     q_values = self.q_net.get_q_values(state_t)
                     action = q_values.argmax(1).item()

                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                original_episode_reward += reward
                state = self.preprocessor.step(next_obs)
                episode_steps += 1
            episode_rewards.append(original_episode_reward)
            # print(f"  - Eval Episode {i+1}: Orig Reward {original_episode_reward:.2f}") # Optional verbose log

        avg_reward = numpy.mean(episode_rewards)
        eval_duration = time.time() - eval_start_time
        print(f"Evaluation finished. Average Original Reward: {avg_reward:.4f} over {num_eval_episodes} episodes. Duration: {eval_duration:.2f}s")

        # --- Cleanup after evaluation (Keep from previous version) ---
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        # --------------------------------

        self.q_net.train() # Set back to training mode
        return avg_reward

    def train(self):
        """ Perform a single training update using C51 loss. """
        # Start training only when buffer has enough samples
        if len(self.memory) < self.replay_start_size:
            # We also check this in run loop, but double check here
            return

        # Sample batch from PER
        sample_result = self.memory.sample(self.batch_size)
        if sample_result is None: return # Sampling failed

        states_np, actions_t, rewards_t, next_states_np, dones_t, weights_t, tree_indices = sample_result

        # Convert states and move all to device
        try:
            states_t = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
            next_states_t = torch.as_tensor(next_states_np, dtype=torch.float32, device=self.device)
            actions_t = actions_t.to(self.device)
            rewards_t = rewards_t.to(self.device)
            dones_t = dones_t.to(self.device)
            weights_t = weights_t.to(self.device)
        except Exception as e:
            print(f"Error during tensor conversion/move: {e}")
            return

        # --- Calculate C51 Loss ---
        # Get current distribution logits for selected actions
        current_logits = self.q_net(states_t) # [B, A, N]
        log_p_current = torch.log_softmax(current_logits, dim=2)
        # Gather the log probabilities for the actions taken
        log_p_a_current = log_p_current[torch.arange(self.batch_size), actions_t] # [B, N]

        # Calculate target distribution (using Double DQN logic)
        with torch.no_grad():
            # Select best actions for next states using ONLINE network's Q-values
            next_q_values = self.q_net.get_q_values(next_states_t) # [B, A]
            next_actions = next_q_values.argmax(1) # [B]

            # Get next state distributions from TARGET network
            target_logits_next = self.target_net(next_states_t) # [B, A, N]
            target_p_next = torch.softmax(target_logits_next, dim=2)
            # Gather the target distributions for the selected next actions
            target_p_next_a = target_p_next[torch.arange(self.batch_size), next_actions] # [B, N]

            # Compute the projected target distribution 'm'
            gamma_n = self.gamma ** self.n_step # N-step discount
            # Ensure support is on the correct device
            support = self.support.unsqueeze(0).expand(self.batch_size, N_ATOMS) # [B, N]

            # Compute target atom values Tz = R + gamma^n * z'
            Tz = rewards_t.unsqueeze(1) + (1 - dones_t.unsqueeze(1)) * gamma_n * support # [B, N]
            Tz = Tz.clamp(V_MIN, V_MAX) # Clamp to support range

            # Project onto support grid
            b = (Tz - V_MIN) / DELTA_Z
            l = b.floor().long()
            u = b.ceil().long()
            # Ensure indices are within bounds [0, N_ATOMS-1]
            l.clamp_(0, N_ATOMS - 1)
            u.clamp_(0, N_ATOMS - 1)

            # Distribute probability mass (dL = u - b, dU = b - l)
            m = torch.zeros_like(target_p_next_a) # [B, N]
            # Use index_add_ for efficient scattering (requires 1D indexing)
            offset = torch.arange(self.batch_size, device=self.device) * N_ATOMS
            m.view(-1).index_add_(0, (l + offset.unsqueeze(1)).view(-1), (target_p_next_a * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset.unsqueeze(1)).view(-1), (target_p_next_a * (b - l.float())).view(-1))
            # m now holds the projected target distribution for each batch item

        # Calculate KL divergence loss (element-wise)
        # loss = sum(target_dist * (log(target_dist) - log(current_dist)))
        # We use cross-entropy: sum(target_dist * -log(current_dist))
        # Ensure m has no zeros before log (add small epsilon)? Softmax output should be > 0.
        # log_p_a_current should be safe due to log_softmax.
        elementwise_loss = -(m * log_p_a_current).sum(1) # [B]

        # Apply PER Importance Sampling weights and calculate mean loss
        loss = (weights_t * elementwise_loss).mean()

        # --- Gradient Descent ---
        self.optimizer.zero_grad()
        loss.backward()
        # Use gradient clipping
        clip_value = getattr(self.args, "gradient_clip_value", 10.0) # Get from args
        clip_grad_norm_(self.q_net.parameters(), clip_value)
        self.optimizer.step()

        # --- Update PER priorities ---
        # Use the element-wise loss (before weighting) as the error for priority update
        td_errors_for_priority = elementwise_loss.detach() # Detach from graph
        self.memory.update_priorities(tree_indices, td_errors_for_priority)

        # --- Soft Target Network Update ---
        # Perform soft update *after* optimization step
        with torch.no_grad():
            for p_online, p_target in zip(self.q_net.parameters(), self.target_net.parameters()):
                p_target.data.mul_(1.0 - self.soft_tau)
                p_target.data.add_(self.soft_tau * p_online.data)

        # --- Logging (Optional - Moved outside train() for clarity, e.g., in run() loop) ---
        # Log stats less frequently to avoid overhead
        log_freq_train = getattr(self.args, "train_log_frequency", 1000) # Log every 1000 *train* steps
        if self.train_count % log_freq_train == 0:
             with torch.no_grad():
                  q_values_mean = self.q_net.get_q_values(states_t).mean().item()
                  # Log loss, Q-mean, TD error mean etc.
                  wandb.log({
                       "Loss/Train Loss": loss.item(),
                       "Stats/Q-Value Mean (Train Batch)": q_values_mean,
                       "Stats/TD_error_mean": td_errors_for_priority.abs().mean().item(),
                       "Stats/Buffer_Coverage": len(self.memory) / self.memory.capacity,
                  }, step=self.env_count) # Log against env steps

        self.train_count += 1 # Increment train count


# --- Helper function for boolean args ---
def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

# --- Environment Creation Helper ---
def make_env(env_name: str, seed: int = None, render_mode: str = "rgb_array"):
    """ Creates the Atari environment with sticky actions. """
    # Explicitly set repeat_action_probability for clarity, use default 0.25
    env = gym.make(env_name, render_mode=render_mode, repeat_action_probability=0.25)
    print(f"Created env {env_name} with repeat_action_probability=0.25")
    # Seeding done via reset
    return env

# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fast & Stable C51 DQN Agent Training")

    # --- Core Training Parameters ---
    parser.add_argument("--train-per-step", type=int, default=4, help="Number of training updates per environment step (default: 4)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for Adam optimizer (recommend lower for high train_per_step, e.g., 2.5e-5 to 5e-5) (default: 5e-5)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training (recommend 64 or 128) (default: 64)")
    parser.add_argument("--soft-tau", type=float, default=1e-3, help="Base tau for soft target updates (actual tau = soft_tau / train_per_step) (default: 1e-3)")
    # parser.add_argument("--target-update-frequency", type=int, default=8000, help="[DEPRECATED by soft updates] Frequency (train steps) for hard target updates")

    # --- Environment Arguments ---
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5", help="Gym environment ID (default: ALE/Pong-v5)")
    parser.add_argument("--seed", type=int, default=777, help="Random seed (default: 777)")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames to stack (default: 4)")
    parser.add_argument("--max-episode-steps", type=int, default=27000, help="Maximum steps per episode (optional limit) (default: 27000 from ALE)") # Default ALE limit

    # --- Training Duration ---
    parser.add_argument("--episodes", type=int, default=2000, help="Total training episodes (adjust based on steps) (default: 2000)")
    parser.add_argument("--total-train-steps", type=int, default=200000, help="Estimated total *environment* steps for beta annealing and potentially epsilon decay (aim for ~100k goal) (default: 200k)")

    # --- Replay Buffer Arguments ---
    parser.add_argument("--memory-size", type=int, default=100000, help="Capacity of the replay buffer (recommend 100k+) (default: 100k)")
    parser.add_argument("--replay-start-size", type=int, default=20000, help="Min env steps before training starts (recommend lower with expert data) (default: 20k)")

    # --- PER Arguments ---
    parser.add_argument("--alpha", type=float, default=0.5, help="PER alpha (default: 0.5)")
    parser.add_argument("--beta", type=float, default=0.4, help="PER initial beta (default: 0.4)")
    parser.add_argument("--reward-scale", type=float, default=1.0, help="Reward scale factor for PER priority (default: 1.0)")

    # --- N-step Learning ---
    parser.add_argument("--n-step", type=int, default=3, help="N-step return calculation (default: 3)")

    # --- C51 Arguments (Keep defaults, V_MIN/V_MAX adjusted globally) ---
    # parser.add_argument("--v-min", type=float, default=V_MIN) # Defined globally
    # parser.add_argument("--v-max", type=float, default=V_MAX) # Defined globally
    # parser.add_argument("--atom-size", type=int, default=N_ATOMS) # Defined globally

    # --- Optimizer ---
    parser.add_argument("--adam-eps", type=float, default=1.5e-4, help="Adam epsilon for stability (default: 1.5e-4, similar to Dopamine)")
    parser.add_argument("--gradient-clip-value", type=float, default=10.0, help="Gradient clipping value (default: 10.0)")

    # --- Epsilon-Greedy (Linear Decay over Env Steps) ---
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-decay-steps", type=int, default=50000, help="Env steps to decay epsilon over (e.g., to reach final in 100k steps) (default: 100k)")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Final epsilon value (default: 0.01)")

    # --- Expert Data ---
    parser.add_argument("--load-expert-data", type=str, default=None, help="Path to expert experience .pkl file to pre-load into PER.")

    # --- Logging and Saving ---
    parser.add_argument("--save-dir", type=str, default="./c51_fast_results", help="Directory for results")
    parser.add_argument("--wandb-project-name", type=str, default="DLP-Lab5-C51-Fast", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--eval-frequency-episodes", type=int, default=10, help="Evaluation frequency (episodes) (default: 10)")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes for each evaluation run (default: 10)")
    parser.add_argument("--save-interval", type=int, default=100000, help="Snapshot saving frequency (env steps) (default: 100k)")
    parser.add_argument("--train-log-frequency", type=int, default=1000, help="Log training stats frequency (train steps) (default: 1000)")
    parser.add_argument("--discount-factor", type=float, default=0.99, help="Discount factor (gamma) (default: 0.99)")


    args = parser.parse_args()

    # Calculate Linear Epsilon Decay Rate (per environment step)
    if args.epsilon_decay_steps > 0:
        args.epsilon_decay_rate = (args.epsilon_start - args.epsilon_min) / args.epsilon_decay_steps
    else:
        args.epsilon_decay_rate = 0
    print(f"Linear Epsilon Decay: start={args.epsilon_start}, final={args.epsilon_min}, steps={args.epsilon_decay_steps}, rate={args.epsilon_decay_rate:.8f}")


    # Setup Wandb Run Name
    if args.wandb_run_name is None:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        # Example name: C51-N3-f4-lr5e-5-tps4-b64-expert_timestamp
        expert_tag = "_expert" if args.load_expert_data else ""
        args.wandb_run_name = f"C51-N{args.n_step}-f{args.frame_stack}-lr{args.lr}-tps{args.train_per_step}-b{args.batch_size}{expert_tag}_{timestamp}"

    # Initialize Wandb
    try:
        wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_run_name,
            config=vars(args),
            save_code=False, # Disable saving code to wandb
            settings=wandb.Settings(_disable_stats=True) # Disable detailed sys stats
        )
        wandb.watch(None) # Disable automatic model watching
        print(f"Wandb initialized for run: {args.wandb_run_name}")
    except Exception as e:
        print(f"Wandb initialization failed: {e}. Training without wandb logging.")

    # Create and run agent
    agent = DQNAgent(env_name=args.env_name, args=args)
    print(f"Starting training for {args.episodes} episodes (aiming for ~{args.total_train_steps} env steps)...")
    agent.run(args.episodes)

    # Final cleanup
    print("Training finished. Cleaning up...")
    if wandb.run is not None: wandb.finish()
    print("Script finished.")