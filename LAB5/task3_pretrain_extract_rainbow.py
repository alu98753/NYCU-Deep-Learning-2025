# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.nn.functional as F # <<< Import F for functional operations
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
from torch.nn.utils import clip_grad_norm_
# -*- coding: utf-8 -*-
"""Segment tree for Prioritized Replay Buffer."""

import operator
from typing import Callable
import pickle

gym.register_envs(ale_py)
# 確保每個子程序設定合理
torch.set_num_threads(4)
torch.set_num_interop_threads(4)  # 也設定 inter-op thread 數量
cv2.setNumThreads(0)  # OpenCV 禁止多執行緒，避免偷吃CPU

# 禁止多餘的 log（選擇性加）
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
# ... (AtariPreprocessor, SumTree, PrioritizedReplayBuffer remain unchanged) ...
# --- AtariPreprocessor ---
class AtariPreprocessor:
    """
    Preprocessing the state input of DQN for Atari
    (Modified for consistency)
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        # Ensure input is numpy array
        if not isinstance(obs, np.ndarray):
             obs = np.array(obs)

        # Check if already grayscale (sometimes env might return grayscale)
        if len(obs.shape) == 2:
            gray = obs
        elif len(obs.shape) == 3 and obs.shape[2] == 1:
             gray = obs.squeeze(axis=2)
        elif len(obs.shape) == 3:
             gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unexpected observation shape: {obs.shape}")

        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8) # Ensure uint8

    def reset(self, obs):
        frame = self.preprocess(obs)
        # No need for 1D check in Atari
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0) # Shape: (4, 84, 84)

    def step(self, obs):
        frame = self.preprocess(obs)
        # No need for 1D check in Atari
        self.frames.append(frame)
        return np.stack(self.frames, axis=0) # Shape: (4, 84, 84)

# --- SumTree ---
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

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

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        if idx != 0: # Avoid propagating from root
             self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        # Ensure dataIdx is valid before accessing self.data
        if dataIdx < 0 or dataIdx >= self.capacity or dataIdx >= self.n_entries:
             dataIdx = np.clip(dataIdx, 0, self.n_entries - 1)
             # <<< Be cautious about printing warnings frequently during high-throughput training >>>
             # print(f"Warning: Invalid data index {idx - self.capacity + 1} retrieved. Clamping to {dataIdx}.")
        return (idx, self.tree[idx], self.data[dataIdx])

# --- PrioritizedReplayBuffer ---
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, reward_scale=1.0, beta_increment_per_sampling=0.001):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6 # Small positive constant for priority
        self.reward_scale = reward_scale # NOTE: Less relevant for C51 rewards directly, but used for scaling errors initially
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.max_priority = 1.0 # <<< Add max_priority tracking for new transitions

    def __len__(self):
        return self.tree.n_entries

    def add(self, transition, error=None): # <<< Modify: error is optional, use max priority if None
        # <<< For C51, we update priorities later based on loss. Add initially with max priority. >>>
        # priority = (abs(error / self.reward_scale) + self.epsilon) ** self.alpha if error is not None else self.max_priority
        priority = self.max_priority # <<< Add all new transitions with max priority
        self.tree.add(priority, transition)

    def sample(self, batch_size):
        batch = []
        idxs = []
        priorities = []
        segment = self.tree.total() / batch_size

        # Anneal beta
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            if self.tree.total() == 0:
                return None, None, None, None, None, None, None
            s = random.uniform(a, b)
            s = min(s, self.tree.total() - 1e-7) # Avoid exceeding total due to float issues

            try:
                idx, priority, data = self.tree.get(s)
                if data is None:
                    # print(f"Warning: Sampled None data for segment [{a}, {b}], s={s}. Resampling or skipping.")
                    continue # Skip this sample
                # <<< Avoid adding data if it looks invalid (can happen during buffer filling) >>>
                if not isinstance(data, tuple) or len(data) != 5:
                    # print(f"Warning: Sampled invalid data format: {data}. Skipping.")
                    continue
                priorities.append(priority)
                idxs.append(idx)
                batch.append(data)
            except Exception as e:
                 print(f"Error during PER sampling: s={s}, total={self.tree.total()}, segment=[{a},{b}]")
                 print(f"Tree state: n_entries={self.tree.n_entries}, write={self.tree.write}")
                 raise e

        if len(batch) < batch_size:
             # print(f"Warning: Could only sample {len(batch)} transitions out of {batch_size}")
             if not batch:
                 return None, None, None, None, None, None, None

        # Importance sampling weights calculation
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = np.power(self.tree.n_entries * (sampling_probabilities + 1e-10), -self.beta)
        weights /= (weights.max() + 1e-10) # Normalize weights

        # Unpack the batch (S_t, A_t, R_n_step, S_tpn, D_tpn)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to appropriate types (keep states/next_states as numpy for now)
        states_np = np.array(states)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        # Rewards are R_n_step (scalar)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_states_np = np.array(next_states)
        # Dones are D_tpn (scalar 0 or 1)
        dones_tensor = torch.tensor(dones, dtype=torch.float32)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        return states_np, actions_tensor, rewards_tensor, next_states_np, dones_tensor, weights_tensor, idxs

    def update_priorities(self, indices, errors): # errors are now per-sample losses or KL divergences
        priorities = (np.abs(errors) + self.epsilon) ** self.alpha
        for i, idx in enumerate(indices):
            self.tree.update(idx, priorities[i])
        # <<< Update max_priority observed >>>
        self.max_priority = max(self.max_priority, priorities.max())


# --- init_weights ---
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        # <<< Slightly different init for Linear layers in Dueling/C51 might be beneficial >>>
        nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5)) # Kaiming uniform with default PyTorch Linear gain
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(m.bias, -bound, bound)


# --- <<< New Network: DuelingC51DQN >>> ---
class DuelingC51DQN(nn.Module):
    def __init__(self, num_actions, num_atoms=51, vmin=-10, vmax=10):
        super(DuelingC51DQN, self).__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.vmin = vmin
        self.vmax = vmax

        # Calculate C51 support atoms
        self.support = torch.linspace(vmin, vmax, num_atoms)
        self.delta_z = (vmax - vmin) / (num_atoms - 1)

        # Shared convolutional base (same as original DQN)
        self.conv_base = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.flattened_size = 64 * 7 * 7 # 3136

        # Dueling Streams
        # Value Stream V(s) - Outputs distribution (num_atoms logits)
        self.value_stream = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_atoms)
        )

        # Advantage Stream A(s, a) - Outputs distribution for each action (num_actions * num_atoms logits)
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions * self.num_atoms)
        )

        # Register support and delta_z as buffers
        self.register_buffer("support_buf", self.support)
        self.register_buffer("delta_z_buf", torch.tensor(self.delta_z))


    def forward(self, x):
        # Normalize input
        x = x / 255.0
        # Pass through convolutional base
        features = self.conv_base(x)

        # Get value and advantage distributions (logits)
        value_logits = self.value_stream(features) # Shape: (batch, num_atoms)
        advantage_logits = self.advantage_stream(features) # Shape: (batch, num_actions * num_atoms)

        # Reshape for Dueling calculation
        value_logits = value_logits.view(-1, 1, self.num_atoms) # Shape: (batch, 1, num_atoms)
        advantage_logits = advantage_logits.view(-1, self.num_actions, self.num_atoms) # Shape: (batch, num_actions, num_atoms)

        # Combine streams using Dueling formula (on logits)
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        mean_advantage_logits = advantage_logits.mean(1, keepdim=True) # Shape: (batch, 1, num_atoms)
        q_logits = value_logits + advantage_logits - mean_advantage_logits # Shape: (batch, num_actions, num_atoms)

        # We return logits, loss function will handle softmax/log_softmax
        return q_logits

    def get_expected_q_values(self, x):
        """ Helper function to get expected Q-values for action selection """
        q_logits = self.forward(x)
        q_probs = F.softmax(q_logits, dim=2) # Convert logits to probabilities
        # Calculate expectation: sum(probability * support_value)
        expected_q = torch.sum(q_probs * self.support_buf, dim=2) # Use registered buffer
        return expected_q


class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.args = args
        if not getattr(args, 'generate_expert_data', False):
            self.env = gym.make(env_name, frameskip=1)
        else:
            self.env = None

        self.test_env = gym.make(env_name, frameskip=1, render_mode=None)
        self.num_actions = self.test_env.action_space.n

        self.preprocessor = AtariPreprocessor(frame_stack=args.frame_stack)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # --- <<< Instantiate DuelingC51DQN >>> ---
        self.q_net = DuelingC51DQN(
            self.num_actions,
            num_atoms=args.num_atoms,
            vmin=args.vmin,
            vmax=args.vmax
        ).to(self.device)
        self.q_net.apply(init_weights) # Apply weight initialization

        self.target_net = DuelingC51DQN(
            self.num_actions,
            num_atoms=args.num_atoms,
            vmin=args.vmin,
            vmax=args.vmax
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval() # Target network is always in eval mode
        # ------------------------------------------

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4 if 'atari' in args.env_name.lower() else 1e-8) # Use smaller eps for Atari stability

        # --- C51 support buffers for target projection ---
        self.support = self.q_net.support_buf.to(self.device) # Get support from network buffer
        self.delta_z = self.q_net.delta_z_buf.to(self.device)
        self.num_atoms = args.num_atoms
        self.vmin = args.vmin
        self.vmax = args.vmax
        # -----------------------------------------------

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon_start = args.epsilon_start
        self.epsilon_decay_steps = args.epsilon_decay_steps
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.decision_steps = 0
        self.train_count = 0
        self.best_reward = -21.0

        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_decision_step = args.train_per_decision_step
        self.frame_skip = args.frame_skip

        if not getattr(args, 'generate_expert_data', False):
            self.save_dir = os.path.join(
                args.save_dir,
                f"{args.wandb_run_name}_fs{self.frame_skip}_{time.strftime('%Y%m%d-%H%M%S')}",
                f"{env_name.replace('/', '_')}"
            )
            os.makedirs(self.save_dir, exist_ok=True)

            self.memory = PrioritizedReplayBuffer(
                args.memory_size,
                alpha=args.per_alpha,
                beta=args.per_beta0,
                # reward_scale=args.reward_scale, # Less direct impact now
                beta_increment_per_sampling= (1.0 - args.per_beta0) / args.per_beta_steps if args.per_beta_steps > 0 else 0.0
            )

            self.n_step = args.n_step
            self.n_step_buffer = deque(maxlen=self.n_step)
            # self.reward_scale = args.reward_scale # Less direct impact now
        else:
            self.save_dir = None
            self.memory = None
            self.n_step_buffer = None
            self.n_step = 1
            # self.reward_scale = 1.0

    def _get_current_epsilon(self):
         """Calculates the current epsilon based on linear decay."""
         fraction = min(1.0, self.decision_steps / self.epsilon_decay_steps)
         # Use max with epsilon_min first, then apply linear interpolation
         current_epsilon = self.epsilon_start + fraction * (self.epsilon_min - self.epsilon_start)
         return max(self.epsilon_min, current_epsilon)


    def select_action(self, state, use_greedy=False):
        current_epsilon = 0.0 if use_greedy else self._get_current_epsilon()

        if random.random() < current_epsilon:
            return self.test_env.action_space.sample()

        state_np = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # --- <<< Calculate expected Q-values from distribution >>> ---
            expected_q = self.q_net.get_expected_q_values(state_tensor) # Shape: (1, num_actions)
            action = expected_q.argmax().item()
            # ----------------------------------------------------------
        return action

    def _calculate_n_step_info(self):
        """Calculates the N-step return and relevant info from the buffer."""
        # R_n_step: N-step accumulated discounted reward (use RAW rewards here)
        # S_t: Initial state of the N-step sequence
        # A_t: Initial action of the N-step sequence
        # S_tpn: State after N decision steps (N * frame_skip env steps)
        # D_tpn: Done flag after N decision steps

        # <<< IMPORTANT: Store RAW accumulated reward in n_step_buffer for C51 >>>
        # <<< The 'accumulated_reward' stored previously was scaled/clipped >>>
        # <<< Need to modify run() loop to store raw reward in n_step_buffer >>>

        R_n_step = 0.0
        S_t, A_t = self.n_step_buffer[0][:2] # s, a
        current_gamma = 1.0 # Discount factor for the current step's reward

        true_n = len(self.n_step_buffer)

        for i in range(true_n):
            # Assumes buffer stores: (s, a, raw_accum_reward, ns, d)
            s, a, r_accum_raw, ns, d = self.n_step_buffer[i]
            R_n_step += current_gamma * r_accum_raw # Use raw reward
            current_gamma *= (self.gamma ** self.frame_skip) # Discount factor for the sequence of actions
            if d:
                 true_n = i + 1
                 break

        S_tpn = self.n_step_buffer[true_n - 1][3] # State after true_n steps
        D_tpn = self.n_step_buffer[true_n - 1][4] # Done flag after true_n steps

        # The discount for the bootstrap value from S_tpn
        # <<< This is gamma^(true_n * frame_skip), not used directly in C51 calculation >>>
        # <<< The Bellman update for C51 incorporates gamma^N inside the projection >>>

        # Return the scalar accumulated N-step reward (raw), needed for target projection
        return S_t, A_t, R_n_step, S_tpn, D_tpn # R_n_step is scalar float


    def run(self, episodes):
        if getattr(self.args, 'generate_expert_data', False):
             print("Skipping training run because generate_expert_data is set.")
             return

        start_time = time.time()
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            if "Pong" in self.args.env_name:
                 fire_obs, _, _, _, _ = self.env.step(1)
                 state = self.preprocessor.step(fire_obs)

            done = False
            total_raw_reward_episode = 0.0
            # total_scaled_reward_episode = 0.0 # Less relevant
            decision_step_count = 0
            self.n_step_buffer.clear()

            while not done:
                if self.max_episode_steps > 0 and decision_step_count >= self.max_episode_steps:
                    # print(f"Episode {ep} truncated at {decision_step_count} decision steps.")
                    break

                current_decision_step_state = state
                action = self.select_action(current_decision_step_state, use_greedy=False)

                # --- Frame Skip Logic ---
                # accumulated_reward = 0.0 # Scaled/clipped (less relevant now)
                accumulated_raw_reward = 0.0 # Store raw reward for N-step buffer
                frame_done = False
                last_obs = None
                for fs_step in range(self.frame_skip):
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    self.env_count += 1
                    accumulated_raw_reward += reward # Accumulate raw reward

                    # clipped_reward = np.clip(reward, -1.0, 1.0) # Clip for logging? Or remove?
                    # scaled_reward = clipped_reward * self.reward_scale
                    # accumulated_reward += scaled_reward

                    frame_done = terminated or truncated
                    last_obs = next_obs
                    if frame_done:
                        break
                # -----------------------

                if last_obs is None:
                     print(f"Warning: last_obs is None after frame skip loop. Action: {action}, State shape: {current_decision_step_state.shape}")
                     break

                next_state = self.preprocessor.step(last_obs)

                # --- N-step Buffer Handling ---
                # <<< Store RAW accumulated reward >>>
                self.n_step_buffer.append((current_decision_step_state, action, accumulated_raw_reward, next_state, frame_done))
                # ----------------------------

                if len(self.n_step_buffer) >= self.n_step:
                    # --- <<< Add transition to PER buffer (with max priority initially) >>> ---
                    # Calculate N-step info (returns raw accumulated reward R_n_step)
                    S_t, A_t, R_n_step_raw, S_tpn, D_tpn = self._calculate_n_step_info()
                    # Store (S_t, A_t, R_n_step_raw, S_tpn, D_tpn)
                    # Note R_n_step_raw is scalar float
                    self.memory.add((S_t, A_t, R_n_step_raw, S_tpn, D_tpn))
                    # Priority will be updated during training based on loss
                    # --------------------------------------------------------------------

                state = next_state
                done = frame_done
                total_raw_reward_episode += accumulated_raw_reward
                # total_scaled_reward_episode += accumulated_reward
                decision_step_count += 1
                self.decision_steps += 1


                # --- Training Step ---
                if self.decision_steps >= self.replay_start_size: # Use decision_steps for start size
                    for _ in range(self.train_per_decision_step):
                         self.train()
                # ---------------------
                # === Saving model if decision_steps % 200000 == 0 and <= 1000000 ===
                if self.decision_steps > 0 and self.decision_steps % 200000 == 0 and self.decision_steps <= 1000000:
                    save_filename = f"LAB5_313554044_task3_pong{self.decision_steps}.pt"
                    save_path = os.path.join(self.save_dir, save_filename)
                    torch.save(self.q_net.state_dict(), save_path)
                    print(f"Saved intermediate model at step {self.decision_steps}: {save_path}")
                # === End Saving ===


            # --- End of Episode ---
            elapsed_time = time.time() - start_time
            steps_per_sec = self.decision_steps / elapsed_time if elapsed_time > 0 else 0 # Use decision steps/sec

            print(f"[End Ep {ep}] Raw Reward: {total_raw_reward_episode:.2f}, "
                  f"DecSteps: {decision_step_count}, TotalDecSteps: {self.decision_steps}, EnvSteps: {self.env_count}, TrainCount: {self.train_count}, "
                  f"Eps: {self._get_current_epsilon():.4f}, Mem: {len(self.memory)}, Beta: {self.memory.beta:.3f}, SPS(Dec): {steps_per_sec:.0f}")

            if not getattr(self.args, 'generate_expert_data', False):
                log_dict = {
                    "Episode": ep,
                    "Episode/Raw Reward": total_raw_reward_episode,
                    # "Episode/Scaled Reward": total_scaled_reward_episode,
                    "Episode/Decision Steps": decision_step_count,
                    "Progress/Env Steps": self.env_count,
                    "Progress/Decision Steps": self.decision_steps,
                    "Progress/Train Count": self.train_count,
                    "Progress/Replay Buffer Size": len(self.memory),
                    "Parameters/Epsilon": self._get_current_epsilon(),
                    "Parameters/PER Beta": self.memory.beta if self.memory else 0,
                    "Performance/Decision Steps Per Second": steps_per_sec
                }

                # --- Evaluation and Saving --- (Inside training mode check)
                if ep % self.args.eval_interval == 0:
                    eval_raw_reward = self.evaluate()
                    print(f"[Evaluate Ep {ep}] Eval Raw Reward: {eval_raw_reward:.2f}")
                    log_dict["Evaluation/Raw Reward"] = eval_raw_reward

                    if eval_raw_reward >= self.best_reward:
                         if eval_raw_reward > 0:
                             self.best_reward = eval_raw_reward
                             model_path = os.path.join(self.save_dir, f"best_model_ep{ep}_rew{eval_raw_reward:.0f}.pt")
                             torch.save(self.q_net.state_dict(), model_path)
                             print(f"Saved new best model to {model_path}")

                wandb.log(log_dict, step=self.decision_steps)

                if self.decision_steps % self.args.save_frequency == 0 and self.decision_steps > 0:
                    model_path = os.path.join(self.save_dir, f"model_dstep{self.decision_steps}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved model checkpoint to {model_path}")


    def evaluate(self, num_episodes=20):
        """Evaluates the agent's policy or generates expert data."""
        # This function remains largely the same, but uses the modified select_action
        # It will correctly use expected Q-values for greedy selection with the C51 network.
        # ... (Code for evaluate is mostly the same as previous version, just ensure it uses self.select_action(state, use_greedy=True)) ...
        # <<< Ensure evaluate code is consistent with previous version's generation logic >>>
        total_rewards = []
        collected_transitions = {
            'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []
        }
        is_generating_data = getattr(self.args, 'generate_expert_data', False)
        target_transitions = getattr(self.args, 'num_expert_transitions', 0) if is_generating_data else 0
        collected_count = 0

        # Use a separate test preprocessor instance to avoid interference
        test_preprocessor = AtariPreprocessor(frame_stack=self.args.frame_stack)

        if is_generating_data:
            print(f"Starting expert data generation. Target: {target_transitions} transitions.")
            self.q_net.eval() # Ensure model is in eval mode

        # Estimate episodes needed for generation
        approx_steps_per_ep = self.max_episode_steps if self.max_episode_steps > 0 else 1000
        eval_ep_count = num_episodes if not is_generating_data else int(target_transitions * 1.5 // approx_steps_per_ep) + 5 # Add buffer

        for i in range(eval_ep_count):
            if is_generating_data and collected_count >= target_transitions:
                break

            obs, _ = self.test_env.reset()
            state = test_preprocessor.reset(obs) # Use test_preprocessor
            if "Pong" in self.args.env_name:
                 fire_obs, _, _, _, _ = self.test_env.step(1)
                 state = test_preprocessor.step(fire_obs) # Use test_preprocessor

            done = False
            episode_raw_reward = 0.0
            decision_step_count = 0

            while not done:
                 if is_generating_data and collected_count >= target_transitions:
                     done = True
                     break
                 if self.max_episode_steps > 0 and decision_step_count >= self.max_episode_steps:
                     break

                 current_decision_step_state_for_saving = np.array(state, dtype=np.uint8)
                 action = self.select_action(state, use_greedy=True) # Use greedy action

                 accumulated_raw_reward = 0.0
                 frame_done = False
                 last_obs = None
                 for _ in range(self.frame_skip):
                     next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                     accumulated_raw_reward += reward
                     frame_done = terminated or truncated
                     last_obs = next_obs
                     if frame_done:
                         break

                 if last_obs is not None:
                     next_state = test_preprocessor.step(last_obs) # Use test_preprocessor
                     next_state_for_saving = np.array(next_state, dtype=np.uint8)
                 else:
                     print("Warning: last_obs is None during evaluation/generation. Ending episode.")
                     frame_done = True
                     next_state = state
                     next_state_for_saving = current_decision_step_state_for_saving

                 if is_generating_data and last_obs is not None: # Only save if step was valid
                     collected_transitions['states'].append(current_decision_step_state_for_saving)
                     collected_transitions['actions'].append(action)
                     collected_transitions['rewards'].append(accumulated_raw_reward) # Raw accumulated reward for this step
                     collected_transitions['next_states'].append(next_state_for_saving)
                     collected_transitions['dones'].append(1 if frame_done else 0)
                     collected_count += 1
                     if collected_count % 1000 == 0:
                         print(f"Collected {collected_count} / {target_transitions} expert transitions...")

                 state = next_state
                 done = frame_done
                 episode_raw_reward += accumulated_raw_reward
                 decision_step_count += 1

            if not is_generating_data:
                total_rewards.append(episode_raw_reward)

        if is_generating_data:
            print(f"\nFinished data generation phase. Collected {collected_count} transitions.")
            if collected_count >= target_transitions:
                # Trim excess and save
                indices = list(range(collected_count))
                random.shuffle(indices) # Shuffle before saving is good practice
                indices = indices[:target_transitions]

                states_np = np.array([collected_transitions['states'][i] for i in indices], dtype=np.uint8)
                actions_np = np.array([collected_transitions['actions'][i] for i in indices], dtype=np.int64)
                rewards_np = np.array([collected_transitions['rewards'][i] for i in indices], dtype=np.float32)
                next_states_np = np.array([collected_transitions['next_states'][i] for i in indices], dtype=np.uint8)
                dones_np = np.array([collected_transitions['dones'][i] for i in indices], dtype=np.uint8)

                output_file = self.args.expert_data_output
                np.savez_compressed(output_file,
                                    states=states_np,
                                    actions=actions_np,
                                    rewards=rewards_np,
                                    next_states=next_states_np,
                                    dones=dones_np)
                print(f"Successfully saved {len(states_np)} shuffled expert transitions to {output_file}")
            else:
                print(f"Warning: Collected only {collected_count} transitions, less than target {target_transitions}.")
            return {} # Return empty dict in generation mode

        return np.mean(total_rewards) if total_rewards else -21.0

    # --- <<< train method with C51 logic >>> ---
    def train(self):
        if self.memory is None or len(self.memory) < self.batch_size or self.decision_steps < self.replay_start_size:
             return # Skip if buffer not ready or in generation mode

        # Sample N-step transitions (s_t, a_t, r_n, s_tpn, d_tpn)
        states_np, actions, rewards_n, next_states_np, dones_n, weights, indices = self.memory.sample(self.batch_size)
        if states_np is None:
            return # Sampling failed

        # Convert numpy states to tensors
        states = torch.from_numpy(states_np.astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(next_states_np.astype(np.float32)).to(self.device)
        # actions, rewards_n, dones_n, weights are already tensors on CPU, move to device
        actions = actions.to(self.device)         # Shape: (batch,)
        rewards_n = rewards_n.to(self.device)     # Shape: (batch,) - N-step scalar raw rewards
        dones_n = dones_n.to(self.device)         # Shape: (batch,) - Done flag after N steps
        weights = weights.to(self.device)         # Shape: (batch,) - PER IS weights

        # --- Calculate Target Distribution ---
        with torch.no_grad():
            # Get next state action logits/probs from target network
            target_next_logits = self.target_net(next_states) # Shape: (batch, num_actions, num_atoms)
            target_next_probs = F.softmax(target_next_logits, dim=2)

            # Use online network to select best next actions (Double DQN)
            # Calculate expected Q-values from online network's next state predictions
            online_next_expected_q = self.q_net.get_expected_q_values(next_states) # Shape: (batch, num_actions)
            next_best_actions = online_next_expected_q.argmax(1) # Shape: (batch,)

            # Get target network's distribution for these best actions
            target_next_best_action_probs = target_next_probs.gather(
                1, next_best_actions.view(-1, 1, 1).expand(-1, 1, self.num_atoms)
            ).squeeze(1) # Shape: (batch, num_atoms)

            # Compute projected Bellman target atoms Tz = R_n + gamma^N * z
            # Ensure rewards_n and dones_n have correct shape for broadcasting
            rewards_n_exp = rewards_n.view(-1, 1) # Shape: (batch, 1)
            dones_n_exp = dones_n.view(-1, 1)     # Shape: (batch, 1)
            support_exp = self.support.view(1, -1) # Shape: (1, num_atoms)

            # Calculate discount factor gamma^N where N is n_step * frame_skip
            n_step_gamma = self.gamma ** (self.n_step * self.frame_skip)

            Tz = rewards_n_exp + (1 - dones_n_exp) * n_step_gamma * support_exp # Shape: (batch, num_atoms)
            Tz = Tz.clamp(self.vmin, self.vmax) # Clip within support bounds

            # --- Projection onto original support ---
            b = (Tz - self.vmin) / self.delta_z  # Shape: (batch, num_atoms)
            l = b.floor().long()
            u = b.ceil().long()

            # Ensure indices are valid BEFORE calculating weights or masks
            l.clamp_(0, self.num_atoms - 1)
            u.clamp_(0, self.num_atoms - 1)

            # Create masks for l==u and l!=u cases
            eq_mask = (l == u)
            ne_mask = ~eq_mask

            # Create target distribution tensor initialized to zeros
            m = torch.zeros_like(target_next_best_action_probs) # Shape: (batch, num_atoms)

            # Prepare offset for index_add_ (or put_)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.num_atoms), self.batch_size).unsqueeze(1).expand(self.batch_size, self.num_atoms).long().to(self.device)

            # === Handle l != u case: Distribute probability ===
            if ne_mask.any(): # Check if there are any l!=u cases
                # Calculate weights for l != u case
                p_l = (target_next_best_action_probs * (u.float() - b))[ne_mask]
                p_u = (target_next_best_action_probs * (b - l.float()))[ne_mask]

                # Get indices and add probabilities
                l_offset_ne = (l + offset)[ne_mask]
                u_offset_ne = (u + offset)[ne_mask]
                m.view(-1).index_add_(0, l_offset_ne.view(-1), p_l.view(-1))
                m.view(-1).index_add_(0, u_offset_ne.view(-1), p_u.view(-1))

            # === Handle l == u case: Assign full probability to index l ===
            if eq_mask.any(): # Check if there are any l==u cases
                # Get indices and probabilities where l == u
                l_offset_eq = (l + offset)[eq_mask]
                p_eq = target_next_best_action_probs[eq_mask]
                m.view(-1).index_add_(0, l_offset_eq.view(-1), p_eq.view(-1))

            target_distribution = m # Shape: (batch, num_atoms)
            # ------------------------------------

        # --- Calculate Loss ---
        # Get current state action distribution logits from online network
        q_logits = self.q_net(states) # Shape: (batch, num_actions, num_atoms)
        # Gather the logits for the actions actually taken
        action_logits = q_logits.gather(1, actions.view(-1, 1, 1).expand(-1, 1, self.num_atoms)).squeeze(1) # Shape: (batch, num_atoms)

        # Calculate cross-entropy loss between predicted logits and target distribution
        # Use log_softmax on predicted logits + negative log likelihood loss (equivalent to cross-entropy)
        log_pred_probs = F.log_softmax(action_logits, dim=1)
        elementwise_loss = -(target_distribution * log_pred_probs).sum(1) # Shape: (batch,)
        loss = (weights * elementwise_loss).mean()

        # Apply PER IS weights
        loss = (weights * elementwise_loss).mean()
        # --------------------

        # --- Optimization ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.args.grad_clip)
        self.optimizer.step()
        # ------------------

        # --- Update PER Priorities ---
        # Use absolute loss (or KL divergence) for priorities
        new_priorities = elementwise_loss.abs().detach().cpu().numpy()
        self.memory.update_priorities(indices, new_priorities)
        # ---------------------------

        self.train_count += 1

        # --- Target Network Update ---
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            # print(f"--- Updated Target Network at Train Step {self.train_count} ---") # Less frequent printing
        # ---------------------------

        # --- Logging ---
        if self.train_count % 100 == 0:
            if not getattr(self.args, 'generate_expert_data', False):
                # Calculate expected Q for logging
                with torch.no_grad():
                    q_probs = F.softmax(action_logits, dim=1)
                    q_exp = (q_probs * self.support).sum(1)
                wandb.log({
                    "Training/Loss": loss.item(),
                    "Training/Q Expected Mean": q_exp.mean().item(),
                    "Training/Q Expected Std": q_exp.std().item(),
                    "Training/Raw Loss Mean": elementwise_loss.mean().item(), # Log unweighted loss too
                }, step=self.decision_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ... (Environment Args, Training Loop Args - mostly unchanged) ...
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5", help="Gym environment ID")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--frame-skip", type=int, default=4, help="Number of frames to skip per action")
    parser.add_argument("--max-episode-steps", type=int, default=27000 // 4, help="Max decision steps per episode")

    parser.add_argument("--episodes", type=int, default=8000 , help="Total number of episodes to train for") # May need more
    parser.add_argument("--replay-start-size", type=int, default=40000, help="Decision steps before starting training")
    parser.add_argument("--train-per-decision-step", type=int, default=1, help="Gradient updates per decision step")

    # --- Algorithm Args ---
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate for Adam optimizer (Adjusted for C51/Atari)") # Often lower for C51
    parser.add_argument("--discount-factor", type=float, default=0.99, help="Discount factor gamma")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--target-update-frequency", type=int, default=2000, help="Frequency (training steps) to update target network (Adjusted for C51)") # Often 8k steps or more
    parser.add_argument("--grad-clip", type=float, default=10.0, help="Gradient clipping value (max norm)")

    # --- <<< C51 Args >>> ---
    parser.add_argument("--num-atoms", type=int, default=51, help="Number of atoms in the C51 distributional representation")
    parser.add_argument("--vmin", type=float, default=-10.0, help="Minimum value for C51 support (Pong rewards are {-1, 0, 1}, but value can exceed this)") # Adjust based on expected N-step return range
    parser.add_argument("--vmax", type=float, default=10.0, help="Maximum value for C51 support") # Adjust based on expected N-step return range
    # -----------------------

    # --- Exploration Args ---
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon value")
    parser.add_argument("--epsilon-min", type=float, default=0.01, help="Minimum epsilon value") # Might go lower (e.g., 0.001) for longer training
    parser.add_argument("--epsilon-decay-steps", type=int, default=250000, help="Decision steps for epsilon linear decay (Longer for C51)") # Rainbow uses 1M steps

    # --- Replay Buffer / PER Args ---
    parser.add_argument("--memory-size", type=int, default=100000, help="Capacity of the replay buffer") # Rainbow uses 1M
    parser.add_argument("--per-alpha", type=float, default=0.5, help="Alpha exponent for PER priorities (Rainbow uses 0.5)")
    parser.add_argument("--per-beta0", type=float, default=0.4, help="Initial beta exponent for PER IS weights (Rainbow uses 0.4)")
    parser.add_argument("--per-beta-steps", type=int, default=500000, help="Training steps to anneal beta to 1.0 (Longer)") # Correlate with total training steps

    # --- N-step Args ---
    parser.add_argument("--n-step", type=int, default=3, help="Number of decision steps for N-step returns")

    # --- Logging and Saving Args ---
    parser.add_argument("--save-dir", type=str, default="./expert_results", help="Directory to save models/logs")
    parser.add_argument("--wandb-project", type=str, default="DLP-Lab5-DuelingC51-Pong", help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default="dueling_c51_run", help="Base WandB run name")
    parser.add_argument("--eval-interval", type=int, default=20, help="Frequency (episodes) for evaluation") # Evaluate less often
    parser.add_argument("--save-frequency", type=int, default=100000, help="Frequency (decision steps) to save checkpoints")
    parser.add_argument("--seed", type=int, default=777, help="Random seed")

    # --- Expert Data Generation Args ---
    parser.add_argument("--generate-expert-data", action='store_true', help="Run in expert data generation mode.")
    parser.add_argument("--load-expert-model", type=str, default=None, help="Path to pre-trained model for data generation.")
    parser.add_argument("--expert-data-output", type=str, default="expert_data_c51.npz", help="Filename for saving generated expert data.")
    parser.add_argument("--num-expert-transitions", type=int, default=50000, help="Number of expert transitions to collect.")

    args = parser.parse_args()

    # --- Argument Validation/Adjustment ---
    if not args.generate_expert_data:
        args.replay_start_size = max(args.batch_size * args.n_step, args.replay_start_size) # Ensure enough steps for n-step buffer fill
        args.target_update_frequency = max(1, args.target_update_frequency)
        args.per_beta_steps = max(0, args.per_beta_steps)
        # Dynamic WandB name
        args.wandb_run_name = f"DuelC51_fs{args.frame_skip}_n{args.n_step}_lr{args.lr}_b{args.batch_size}_atoms{args.num_atoms}_v{args.vmin}-{args.vmax}_seed{args.seed}"
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args), save_code=True)
    # --- Vmin/Vmax validation ---
    if args.vmin >= args.vmax:
        raise ValueError("--vmin must be strictly less than --vmax")
    # --------------------------

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Consider benchmark=True for speed if input sizes are fixed
        # torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True # For strict reproducibility if needed

    # Instantiate agent
    agent = DQNAgent(env_name=args.env_name, args=args)

    # --- Execute Mode ---
    if args.generate_expert_data:
        print("--- Running in Expert Data Generation Mode ---")
        if args.load_expert_model is None or not os.path.exists(args.load_expert_model):
            raise ValueError("--load-expert-model must be specified and exist in generation mode.")
        print(f"Loading expert model weights from: {args.load_expert_model}")
        # Load weights carefully, potentially ignoring size mismatches if loading older model?
        # For now, assume the loaded model matches DuelingC51DQN architecture.
        agent.q_net.load_state_dict(torch.load(args.load_expert_model, map_location=agent.device))
        print("Expert model loaded successfully.")
        agent.evaluate(num_episodes=-1) # Run generation via evaluate
        print("--- Expert Data Generation Finished ---")
    else:
        print("--- Running in Training Mode ---")
        agent.run(args.episodes)
        wandb.finish()
        print("--- Training Finished ---")