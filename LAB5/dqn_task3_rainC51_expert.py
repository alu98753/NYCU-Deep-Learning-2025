# -*- coding: utf-8 -*-
import math
import os
import time
import random
import pickle
import argparse
from collections import deque
import operator
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

import gymnasium as gym
import ale_py
import wandb
import cv2
from torch.amp import GradScaler # Keep import even if disabled
from torch.nn.utils import clip_grad_norm_
torch.backends.cudnn.benchmark = True

gym.register_envs(ale_py)
cv2.setNumThreads(0)

# Distributional RL hyperparams
N_ATOMS = 51
V_MIN, V_MAX = -10.0, 10.0
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)
SUPPORT_CPU = torch.linspace(V_MIN, V_MAX, N_ATOMS)

# ----------------------------------------------------------------------------
# Segment trees for PER (No changes from previous version)
# ----------------------------------------------------------------------------
class SegmentTree:
    def __init__(self, capacity: int, operation: Callable, init_value: float):
        assert capacity > 0 and capacity & (capacity - 1) == 0
        self.capacity = capacity
        self.tree = [init_value] * (2 * capacity)
        self.operation = operation

    def _operate(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate(start, end, 2*node, node_start, mid)
        if start > mid:
            return self._operate(start, end, 2*node+1, mid+1, node_end)
        return self.operation(
            self._operate(start, mid, 2*node, node_start, mid),
            self._operate(mid+1, end, 2*node+1, mid+1, node_end)
        )

    def operate(self, start=0, end=0):
        if end <= 0: end += self.capacity
        end = min(end, self.capacity)
        # Return neutral element (or initial value) if range is invalid/empty
        if start >= end: return self.tree[0] if self.operation == operator.add else float('inf') if self.operation == min else 0
        end -= 1
        return self._operate(start, end, 1, 0, self.capacity-1)

    def __setitem__(self, idx, val):
        idx += self.capacity
        self.tree[idx] = val
        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2*idx], self.tree[2*idx+1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self.capacity
        return self.tree[self.capacity + idx]

class SumSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity, operation=operator.add, init_value=0.0)

    def sum(self, start=0, end=0):
        if end == 0: end = self.capacity
        return super().operate(start, end)

    def retrieve(self, upperbound: float):
        idx = 1
        while idx < self.capacity:
            left = 2*idx
            if self.tree[left] > upperbound:
                idx = left
            else:
                upperbound -= self.tree[left]
                idx = left + 1
        return idx - self.capacity

class MinSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity, operation=min, init_value=float('inf'))
    def min(self, start=0, end=0):
        if end == 0: end = self.capacity
        return super().operate(start, end)

# ----------------------------------------------------------------------------
# Replay buffers (Re-enabled Pinned Memory)
# ----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, obs_shape, size, batch_size=32, n_step=1, gamma=0.99):
        # --- FIX: Re-enable Pinned Memory ---
        self.obs_buf = torch.zeros((size, *obs_shape), dtype=torch.uint8)
        self.next_obs_buf = torch.zeros((size, *obs_shape), dtype=torch.uint8)
        self.acts_buf = torch.zeros(size, dtype=torch.int64)
        self.rews_buf = torch.zeros(size, dtype=torch.float32)
        self.done_buf = torch.zeros(size, dtype=torch.float32)
        # --- End Fix ---

        self.max_size = size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self.n_step = n_step
        self.gamma = gamma
        self.n_buffer = deque(maxlen=n_step)

    def store(self, obs, act, rew, next_obs, done):
        self.n_buffer.append((obs, act, rew, next_obs, done))
        if len(self.n_buffer) < self.n_step:
            return None

        R, Nobs, D = self._get_n_step_info()
        obs0, act0 = self.n_buffer[0][:2]

        self.obs_buf[self.ptr].copy_(torch.from_numpy(obs0))
        self.next_obs_buf[self.ptr].copy_(torch.from_numpy(Nobs))
        self.acts_buf[self.ptr] = act0
        self.rews_buf[self.ptr] = R
        self.done_buf[self.ptr] = float(D)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        return self.n_buffer[0]

    def _get_n_step_info(self):
        R, Nobs, D = self.n_buffer[-1][-3:]
        for (o,a,r,o2,d) in list(self.n_buffer)[:-1][::-1]:
            R = r + self.gamma * R * (1 - d)
            if d: Nobs, D = o2, d
        return R, Nobs, D

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, obs_shape, size, batch_size=32, alpha=0.6, n_step=1, gamma=0.99):
        super().__init__(obs_shape, size, batch_size, n_step, gamma)
        self.alpha = alpha
        self.max_prio = 1.0
        cap = 1
        while cap < size: cap *= 2
        self.sum_tree = SumSegmentTree(cap)
        self.min_tree = MinSegmentTree(cap)
        self.capacity = cap # Store tree capacity for checks

    def store(self, obs, act, rew, next_obs, done):
        write_ptr = self.ptr
        super().store(obs, act, rew, next_obs, done)
        if len(self.n_buffer) == self.n_step:
             prio = self.max_prio ** self.alpha
             self.sum_tree[write_ptr] = prio
             self.min_tree[write_ptr] = prio

    def sample_batch(self, beta):
        idxs_list = []
        current_active_size = self.size
        if current_active_size == 0: return None, None

        total_p = self.sum_tree.sum(0, current_active_size)
        if total_p <= 0:
            print("Warning: Total priority sum is zero. Sampling uniformly.")
            idxs_np = np.random.choice(current_active_size, self.batch_size, replace=(current_active_size < self.batch_size))
            idxs = torch.from_numpy(idxs_np).long()
            weights = torch.ones(self.batch_size, dtype=torch.float32)
            return idxs, weights

        segment = total_p / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i+1)
            s = random.uniform(a, b)
             # Clip s just below total_p to handle potential floating point issues
            s = min(s, total_p - 1e-7)
            assert 0 <= s < total_p, f"Sample value s={s} out of range [0, {total_p})"
            idx = self.sum_tree.retrieve(s)
            idxs_list.append(min(idx, current_active_size - 1))

        idxs = torch.tensor(idxs_list, dtype=torch.long) # CPU tensor

        weights = []
        p_min = self.min_tree.min(0, current_active_size) / (total_p + 1e-8)
        max_weight = (p_min * current_active_size + 1e-8) ** (-beta)

        for idx_val in idxs_list:
            p = self.sum_tree[idx_val] / (total_p + 1e-8)
            w = (p * current_active_size + 1e-8) ** (-beta)
            weights.append(w / (max_weight + 1e-8))

        weights_tensor = torch.tensor(weights, dtype=torch.float32) # CPU tensor
        return idxs, weights_tensor

    def update_priorities(self, idxs, prios):
        # idxs, prios expected as CPU tensors
        idxs_list = idxs.tolist()
        prios_list = prios.tolist()

        for i, p in zip(idxs_list, prios_list):
            # --- FIX: Check against buffer's max_size ---
            if not 0 <= i < self.max_size:
            # --- Was: if not 0 <= i < self.capacity: --- # (capacity referred to tree capacity before)
                print(f"Warning: Attempting to update priority for invalid index {i} (max_size: {self.max_size})")
                continue
            assert p > 0, f"Priority must be positive, got {p}"
            p_alpha = p ** self.alpha
            # Use the index i directly, it corresponds to the leaf node
            # in the segment tree due to how tree indices map from buffer indices
            self.sum_tree[i] = p_alpha
            self.min_tree[i] = p_alpha
            self.max_prio = max(self.max_prio, p)

# ----------------------------------------------------------------------------
# Atari preprocessing (No changes)
# ----------------------------------------------------------------------------
class AtariPreprocessor:
    def __init__(self, frame_stack):
        self.stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84,84), interpolation=cv2.INTER_AREA).astype(np.uint8)
    def reset(self, obs):
        f = self.preprocess(obs)
        self.frames = deque([f]*self.stack, maxlen=self.stack)
        return np.stack(self.frames,0)
    def step(self, obs):
        f = self.preprocess(obs)
        self.frames.append(f)
        return np.stack(self.frames,0)

# ----------------------------------------------------------------------------
# C51 network (Modified for Dueling Architecture)
# ----------------------------------------------------------------------------
class C51Network(nn.Module):
    def __init__(self, num_actions, frame_stack=4):
        super().__init__()
        self.num_actions = num_actions
        self.feature = nn.Sequential(
            nn.Conv2d(frame_stack, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            d = torch.zeros(1, frame_stack, 84, 84, device='cpu')
            dim = self.feature(d).shape[1]

        # Shared hidden layer
        self.fc_hidden = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU()
        )

        # Value stream outputting logits for V(s) distribution
        self.value_stream = nn.Linear(512, N_ATOMS)

        # Advantage stream outputting logits for A(s, a) distributions
        self.advantage_stream = nn.Linear(512, num_actions * N_ATOMS)

    def forward(self, x):
        # Input x assumed to be float and normalized
        features = self.feature(x)
        hidden = self.fc_hidden(features) # (B, 512)

        # Calculate Value and Advantage logits
        value_logits = self.value_stream(hidden) # (B, N_ATOMS)
        advantage_logits = self.advantage_stream(hidden) # (B, num_actions * N_ATOMS)

        # Reshape for combination
        # Value: (B, 1, N_ATOMS) - Add action dimension
        value_logits = value_logits.view(-1, 1, N_ATOMS)
        # Advantage: (B, num_actions, N_ATOMS)
        advantage_logits = advantage_logits.view(-1, self.num_actions, N_ATOMS)

        # Combine streams: Q = V + (A - mean(A)) applied to logits
        mean_advantage_logits = advantage_logits.mean(dim=1, keepdim=True) # (B, 1, N_ATOMS)
        adjusted_advantage_logits = advantage_logits - mean_advantage_logits # (B, num_actions, N_ATOMS)
        q_logits = value_logits + adjusted_advantage_logits # (B, num_actions, N_ATOMS)

        # Apply softmax to get distributions
        return F.softmax(q_logits, dim=2) # Softmax over the atoms dimension
    
# ----------------------------------------------------------------------------
# Agent (Updated parts marked)
# ----------------------------------------------------------------------------
class RainbowDQfDAgent:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.support = SUPPORT_CPU.to(self.device) # Use self.support

        self.env = gym.make(args.env_name, render_mode='rgb_array')
        self.env.action_space.seed(self.seed)

        self.gamma = args.gamma
        self.n_step = args.n_step
        self.initial_beta = args.beta
        self.beta_anneal = args.beta_anneal_steps
        self.init_margin = args.supervised_margin
        self.margin_anneal = args.margin_anneal_steps
        self.lambda_sup = args.lambda_supervised
        self.target_update = args.target_update
        self.eval_epsilon = args.eval_epsilon
        self.max_episode_steps = args.max_episode_steps

        self.prep = AtariPreprocessor(args.frame_stack)
        self.frame_skip = args.frame_skip

        obs_shape = (args.frame_stack, 84, 84)
        self.memory = PrioritizedReplayBuffer(obs_shape, args.memory_size,
                                              args.batch_size, alpha=args.alpha,
                                              n_step=self.n_step, gamma=self.gamma)

        self.expert_idxs = set()
        if args.expert_data_path and args.num_expert_transitions > 0:
             print(f"Loading expert data from {args.expert_data_path}...")
             # (Expert data loading logic remains the same)
             try:
                 if not os.path.exists(args.expert_data_path):
                      raise FileNotFoundError(f"Expert data file not found: {args.expert_data_path}")
                 with open(args.expert_data_path, 'rb') as f: expert_data = pickle.load(f)
                 num_to_load = min(len(expert_data), args.num_expert_transitions, self.memory.max_size)
                 print(f"Prefilling buffer with {num_to_load} expert transitions...")
                 for i in range(num_to_load):
                      s, a, r, ns, d = expert_data[i]
                      write_ptr = self.memory.ptr
                      self.memory.obs_buf[write_ptr].copy_(torch.from_numpy(s))
                      self.memory.next_obs_buf[write_ptr].copy_(torch.from_numpy(ns))
                      self.memory.acts_buf[write_ptr] = a
                      self.memory.rews_buf[write_ptr] = r
                      self.memory.done_buf[write_ptr] = float(d)
                      prio = self.memory.max_prio ** self.memory.alpha
                      self.memory.sum_tree[write_ptr] = prio
                      self.memory.min_tree[write_ptr] = prio
                      self.expert_idxs.add(write_ptr)
                      self.memory.ptr = (self.memory.ptr + 1) % self.memory.max_size
                      self.memory.size = min(self.memory.size + 1, self.memory.max_size)
                 print(f"Buffer prefilled. Size: {self.memory.size}/{self.memory.max_size}")
             except FileNotFoundError as e: print(f"Error: {e}. Warning: Continuing without expert data.")
             except Exception as e: print(f"Error loading expert data: {e}. Warning: Continuing without expert data.")
        else: print("Skipping expert data prefilling.")

        self.num_actions = self.env.action_space.n
        self.online = C51Network(self.num_actions, args.frame_stack).to(self.device)
        self.target = C51Network(self.num_actions, args.frame_stack).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=args.lr, eps=1.5e-4)

        def lr_lambda(step):
            if step < args.warmup_steps:
                return float(step) / float(max(1, args.warmup_steps))
            progress = float(step - args.warmup_steps) / float(max(1, args.total_steps - args.warmup_steps))
            progress = min(1.0, progress)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.eps_start, self.eps_end = args.eps_start, args.eps_end
        self.eps_decay = args.eps_decay

        self.wandb = wandb
        self.total_steps = 0
        self.scaler = None # Keep AMP disabled
        print("Mixed Precision Disabled.")

    def beta(self): return min(1.0, self.initial_beta + self.total_steps / self.beta_anneal * (1 - self.initial_beta))
    def margin(self): return max(0, self.init_margin * (1 - self.total_steps / self.margin_anneal))
    def epsilon(self): return max(self.eps_end, self.eps_start - self.total_steps / self.eps_decay)

    def select_action(self, state, eval_mode=False):
        current_eps = self.eval_epsilon if eval_mode else self.epsilon()
        if random.random() < current_eps:
            return self.env.action_space.sample()

        state_tensor = torch.from_numpy(state).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.0
        with torch.no_grad(): # No autocast
            probs = self.online(state_tensor)
            q = (probs * self.support).sum(2) # Use self.support
            return int(q.argmax(1).item())

    @torch.no_grad()
    def projection(self, nxt_p, rew, done):
        B = rew.size(0)
        Tz = rew.unsqueeze(1) + (1 - done.unsqueeze(1)) * ((self.gamma ** self.n_step) * self.support.unsqueeze(0))
        Tz = Tz.clamp(V_MIN, V_MAX)
        b = (Tz - V_MIN) / DELTA_Z
        l = b.floor().long()
        u = b.ceil().long()
        l_eq_u = (l == u)
        p_l_orig = nxt_p * (u.float() - b)
        p_u_orig = nxt_p * (b - l.float())
        p_l = torch.where(l_eq_u, nxt_p, p_l_orig)
        p_u = torch.where(l_eq_u, torch.zeros_like(p_u_orig), p_u_orig)

        offset = torch.arange(0, B * N_ATOMS, N_ATOMS, device=self.device).unsqueeze(1)
        l_clamped = l.clamp(0, N_ATOMS - 1)
        u_clamped = u.clamp(0, N_ATOMS - 1)
        flat_l_idx = (l_clamped + offset).view(-1)
        flat_u_idx = (u_clamped + offset).view(-1)
        flat_p_l = p_l.view(-1)
        flat_p_u = p_u.view(-1)

        m_flat = torch.zeros(B * N_ATOMS, device=self.device)
        m_flat.scatter_add_(0, flat_l_idx, flat_p_l)
        m_flat.scatter_add_(0, flat_u_idx, flat_p_u)
        m = m_flat.view(B, N_ATOMS)
        return m

# Inside class RainbowDQfDAgent:

# Inside class RainbowDQfDAgent:

    def update_model(self):
        if self.memory.size < self.args.batch_size: return None

        current_beta = self.beta()
        # sample_batch returns CPU tensors idxs_tensor (long), weights_tensor (float)
        sample_result = self.memory.sample_batch(current_beta)
        if sample_result is None: return None
        idxs_tensor, weights_tensor = sample_result # <-- Renamed variables

        # --- FIX: Remove unnecessary from_numpy calls ---
        # idxs = torch.from_numpy(np.array(idxs_np)).long() # REMOVE
        # weights = torch.from_numpy(weights_np).float()    # REMOVE
        idxs = idxs_tensor # Is already a Long Tensor on CPU
        weights = weights_tensor # Is already a Float Tensor on CPU
        # --- End Fix ---

        # Transfer weights to GPU
        weights_gpu = weights.to(self.device, non_blocking=False).unsqueeze(1)

        # Get integer indices as list for accessing numpy buffers and set lookup
        idxs_list = idxs.tolist()

        # Get data from NumPy buffers using list indices, create tensors, transfer to GPU
        obs = torch.tensor(self.memory.obs_buf[idxs_list], device=self.device).float() / 255.0
        nxt = torch.tensor(self.memory.next_obs_buf[idxs_list], device=self.device).float() / 255.0
        acts = torch.tensor(self.memory.acts_buf[idxs_list], device=self.device).long().unsqueeze(1)
        rews = torch.tensor(self.memory.rews_buf[idxs_list], device=self.device).float()
        dones = torch.tensor(self.memory.done_buf[idxs_list], device=self.device).float()

        # Target distribution calculation (with Double DQN)
        with torch.no_grad():
            online_next_dist = self.online(nxt)
            online_next_q = (online_next_dist * self.support).sum(2)
            nxt_a = online_next_q.argmax(1)
            target_next_dist_all = self.target(nxt)
            nxt_dist = target_next_dist_all[torch.arange(self.args.batch_size), nxt_a]
            tgt = self.projection(nxt_dist, rews, dones)

        # Loss calculation
        curr_p = self.online(obs)
        curr_pa = curr_p[torch.arange(self.args.batch_size), acts.squeeze(1)]
        dist_loss_per_item = -(tgt * torch.log(curr_pa + 1e-8)).sum(1)

        # Supervised Loss
        current_margin = self.margin()
        Q = (curr_p * self.support).sum(2)
        aexp = acts.squeeze(1)
        Qexp = Q[torch.arange(self.args.batch_size), aexp]
        mloss = torch.clamp(Q + current_margin - Qexp.unsqueeze(1), min=0)
        mloss.scatter_(1, aexp.unsqueeze(1), 0.0)

        # Use integer list indices for set lookup
        mask = torch.tensor([1.0 if i in self.expert_idxs else 0.0 for i in idxs_list],
                            device=self.device, dtype=torch.float32)
        num_expert_samples = mask.sum()
        sup_loss = (mloss.sum(1) * mask).sum() / num_expert_samples if num_expert_samples > 0 else torch.tensor(0.0, device=self.device)

        # Combine losses
        weighted_dist_loss = (dist_loss_per_item * weights_gpu.squeeze(1)).mean()
        loss = weighted_dist_loss + self.lambda_sup * sup_loss

        # Optimization
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.lr_scheduler.step()

        # Update priorities (needs CPU tensors/data)
        prios_cpu = (dist_loss_per_item.detach().cpu() + 1e-6)
        # Pass CPU idxs tensor and CPU prios tensor
        self.memory.update_priorities(idxs, prios_cpu) # idxs is already the CPU tensor

        if self.total_steps % self.target_update == 0:
            self.target.load_state_dict(self.online.state_dict())

        return loss.item(), weighted_dist_loss.item(), sup_loss.item()



    def train(self, num_episodes, eval_interval):
        for ep in range(1, num_episodes + 1):
            obs, _ = self.env.reset(seed=self.seed + ep)
            state = self.prep.reset(obs)
            done = False
            ep_r = 0
            ep_steps = 0
            ep_loss_sum = 0
            ep_dist_loss_sum = 0
            ep_sup_loss_sum = 0
            ep_updates = 0
            start_time = time.time()

            while not done:
                # --- Episode Step Limit ---
                if self.max_episode_steps > 0 and ep_steps >= self.max_episode_steps:
                     # Use a more concise truncation message
                     # print(f"      Episode {ep} truncated at {ep_steps} steps.", flush=True)
                     done = True # Force truncation flag

                if done: break

                # --- Agent-Environment Interaction ---
                a = self.select_action(state)
                tot_r = 0
                frame_done = False # Tracks env termination within frame skip
                last_obs = None
                for _ in range(self.frame_skip):
                    nxt, rew, ter, tr, _ = self.env.step(a)
                    tot_r += rew
                    frame_done = ter or tr # Check for termination or truncation from env
                    last_obs = nxt
                    if frame_done: break
                ns = self.prep.step(last_obs)

                # Store transition using actual env done flag
                self.memory.store(state, a, tot_r, ns, frame_done)

                state = ns
                # Update loop 'done' based on env state OR truncation limit
                done = frame_done or done # done remains True if truncated

                ep_r += tot_r
                self.total_steps += 1
                ep_steps += 1

                # --- Model Update ---
                update_result = self.update_model()
                if update_result is not None:
                     loss_val, dist_loss_val, sup_loss_val = update_result
                     ep_loss_sum += loss_val
                     ep_dist_loss_sum += dist_loss_val
                     ep_sup_loss_sum += sup_loss_val
                     ep_updates += 1

            # --- End of Episode ---
            ep_duration = time.time() - start_time
            avg_loss = ep_loss_sum / ep_updates if ep_updates > 0 else 0
            avg_dist_loss = ep_dist_loss_sum / ep_updates if ep_updates > 0 else 0
            avg_sup_loss = ep_sup_loss_sum / ep_updates if ep_updates > 0 else 0

            print(f"Ep {ep:04d}/{num_episodes} | Steps: {ep_steps:4d} | TotSteps: {self.total_steps:7d} | "
                  f"Reward: {ep_r:6.1f} | Eps: {self.epsilon():.3f} | Beta: {self.beta():.3f} | "
                  f"Marg: {self.margin():.3f} | LR: {self.optimizer.param_groups[0]['lr']:.1e} | "
                  f"Loss: {avg_loss:.3f} (D:{avg_dist_loss:.3f}, S:{avg_sup_loss:.3f}) | "
                  f"Mem: {self.memory.size:6d} | Dur: {ep_duration:.2f}s", flush=True)

            log_data = {
                'Train/Episode Reward': ep_r, 'Train/Episode Length': ep_steps,
                'Train/Episode Duration (s)': ep_duration, 'Train/Average Loss': avg_loss,
                'Train/Average Distributional Loss': avg_dist_loss, 'Train/Average Supervised Loss': avg_sup_loss,
                'Parameters/Epsilon': self.epsilon(), 'Parameters/PER Beta': self.beta(),
                'Parameters/Supervised Margin': self.margin(), 'Parameters/Learning Rate': self.optimizer.param_groups[0]['lr'],
                'Memory/Size': self.memory.size
            }
            self.wandb.log(log_data, step=self.total_steps)

            # --- Evaluation Phase ---
            if ep % eval_interval == 0:
                print(f"--- Evaluating @ Episode {ep} (TotalSteps {self.total_steps}) ---", flush=True)
                eval_rewards = []
                with torch.no_grad(): # Wrap evaluation loop
                    for eval_ep in range(5):
                        eval_obs, _ = self.env.reset(seed=self.seed + ep + eval_ep + 10000)
                        eval_state = self.prep.reset(eval_obs)
                        eval_done = False
                        eval_rsum = 0
                        eval_steps = 0
                        while not eval_done:
                            if self.max_episode_steps > 0 and eval_steps >= self.max_episode_steps:
                                eval_done = True
                            if eval_done: break

                            eval_a = self.select_action(eval_state, eval_mode=True)
                            eval_tot_r = 0
                            eval_frame_done = False
                            eval_last_obs = None
                            for _ in range(self.frame_skip):
                                eval_nxt, eval_rew, eval_ter, eval_tr, _ = self.env.step(eval_a)
                                eval_tot_r += eval_rew
                                eval_frame_done = eval_ter or eval_tr
                                eval_last_obs = eval_nxt
                                if eval_frame_done: break
                            eval_ns = self.prep.step(eval_last_obs)
                            eval_state = eval_ns
                            eval_rsum += eval_tot_r
                            eval_done = eval_frame_done or eval_done
                            eval_steps += 1
                        eval_rewards.append(eval_rsum)

                avg_eval_reward = np.mean(eval_rewards)
                std_eval_reward = np.std(eval_rewards)
                print(f"--- Eval Results @ Ep {ep}: Avg Reward={avg_eval_reward:.2f} +/- {std_eval_reward:.2f} (over {len(eval_rewards)} eps) ---", flush=True)
                self.wandb.log({'Eval/Average Reward': avg_eval_reward,
                                'Eval/Std Reward': std_eval_reward}, step=self.total_steps)

if __name__=='__main__':
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Environment
    p.add_argument('--env_name',default='ALE/Pong-v5', help="Gymnasium environment ID")
    p.add_argument('--seed',type=int,default=777, help="Random seed")
    p.add_argument('--frame_stack',type=int,default=4, help="Number of frames to stack")
    p.add_argument('--frame_skip',type=int,default=4, help="Number of frames to skip per action")
    p.add_argument('--max_episode_steps', type=int, default=27000, help="Max steps per episode (0 for no limit)")

    # Training Loop
    p.add_argument('--num_episodes',type=int,default=5000, help="Total number of training episodes")
    p.add_argument('--total_steps',type=int,default=1000000, help="Total training steps for LR cosine decay schedule")

    # Replay Buffer
    p.add_argument('--memory_size',type=int,default=100000, help="Replay buffer size")
    p.add_argument('--batch_size',type=int,default=32, help="Batch size for training")
    p.add_argument('--n_step',type=int,default=3, help="N-step returns")

    # PER
    p.add_argument('--alpha',type=float,default=0.5, help="PER alpha (prioritization exponent)")
    p.add_argument('--beta',type=float,default=0.4, help="Initial PER beta (importance sampling exponent)")
    p.add_argument('--beta_anneal_steps',type=int,default=1000000, help="Steps to anneal beta to 1.0")

    # DQfD / Supervised Loss
    p.add_argument('--expert_data_path',default='./expert_data_pong.pkl', help="Path to expert demonstration data (.pkl)")
    p.add_argument('--num_expert_transitions',type=int,default=20000, help="Number of expert transitions to prefill")
    p.add_argument('--supervised_margin',type=float,default=0.8, help="Supervised loss margin")
    p.add_argument('--margin_anneal_steps',type=int,default=200000, help="Steps to anneal margin to 0")
    p.add_argument('--lambda_supervised',type=float,default=1.0, help="Weight for supervised loss")

    # DQN Algorithm Details
    p.add_argument('--gamma',type=float,default=0.99, help="Discount factor")
    p.add_argument('--lr',type=float,default=1e-4, help="Learning rate for Adam")
    p.add_argument('--warmup_steps',type=int,default=2000, help="Number of steps for LR warmup")
    p.add_argument('--target_update',type=int,default=8000, help="Frequency (in steps) to update target network")
    p.add_argument('--eps_start',type=float,default=1.0, help="Initial epsilon for exploration")
    p.add_argument('--eps_end',type=float,default=0.05, help="Final epsilon for exploration")
    p.add_argument('--eps_decay',type=int,default=75000, help="Steps over which to decay epsilon linearly")

    # Evaluation
    p.add_argument('--eval_interval',type=int,default=50, help="Frequency (in episodes) to perform evaluation")
    p.add_argument('--eval_epsilon', type=float, default=0.001, help='Epsilon used for evaluation')

    # Logging
    p.add_argument('--wandb_project',default='Rainbow-DQfD', help="WandB project name")
    p.add_argument('--wandb_run_name',default=None, help="WandB run name (defaults to auto-generated)")

    args=p.parse_args()

    if args.wandb_run_name is None:
        env_short_name = args.env_name.split('/')[-1].replace('-v5', '')
        args.wandb_run_name = f"{env_short_name}-DQfD-{int(time.time())}"

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    agent=RainbowDQfDAgent(args)
    agent.train(args.num_episodes, args.eval_interval)
    wandb.finish()
    print("Training finished.")