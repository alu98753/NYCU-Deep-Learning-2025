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
# Segment trees for PER (No changes)
# ----------------------------------------------------------------------------
class SegmentTree:
    # ... (Segment Tree code remains exactly the same as your last version) ...
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
            # Boundary check is good practice here
            if left >= len(self.tree): break
            if self.tree[left] > upperbound:
                idx = left
            else:
                right = left + 1
                # Ensure right doesn't go out of bounds for tree access
                # This logic assumes upperbound is valid for the current sum
                if right >= len(self.tree): break
                upperbound -= self.tree[left]
                idx = right
        final_idx = idx - self.capacity
        # Clamp index to valid range [0, capacity-1]
        return max(0, min(final_idx, self.capacity - 1))


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity: int):
        super().__init__(capacity, operation=min, init_value=float('inf'))
    def min(self, start=0, end=0):
        if end == 0: end = self.capacity
        return super().operate(start, end)

# ----------------------------------------------------------------------------
# Replay buffers (Using NumPy arrays - NO Pinned Memory)
# ----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, obs_shape, size, batch_size=32, n_step=1, gamma=0.99):
        # --- Using NumPy arrays as requested ---
        self.obs_buf = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.next_obs_buf = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.acts_buf = np.zeros(size, dtype=np.int64)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32) # Store as float
        # --- End NumPy ---

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

        # Write to NumPy arrays
        self.obs_buf[self.ptr] = obs0
        self.next_obs_buf[self.ptr] = Nobs
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
        self.capacity = cap

    def store(self, obs, act, rew, next_obs, done):
        write_ptr = self.ptr
        result = super().store(obs, act, rew, next_obs, done)
        # Only update priorities if super().store actually stored something
        if result is not None:
             prio = self.max_prio ** self.alpha
             self.sum_tree[write_ptr] = prio
             self.min_tree[write_ptr] = prio

    def sample_batch(self, beta):
        idxs_list = []
        current_active_size = self.size
        if current_active_size == 0: return None, None

        total_p = self.sum_tree.sum(0, current_active_size)
        if total_p <= 0:
            # Fallback to uniform sampling if priorities invalid
            idxs_np = np.random.choice(current_active_size, self.batch_size, replace=(current_active_size < self.batch_size))
            weights_np = np.ones(self.batch_size, dtype=np.float32) / current_active_size # Uniform weights
            return idxs_np, weights_np

        segment = total_p / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i+1)
            s = random.uniform(a, b)
            s = min(s, total_p - 1e-7)
            assert 0 <= s < total_p
            idx = self.sum_tree.retrieve(s)
            idxs_list.append(min(idx, current_active_size - 1))

        # Return NumPy arrays as buffer uses NumPy
        idxs_np = np.array(idxs_list, dtype=np.int64)

        weights = []
        p_min = self.min_tree.min(0, current_active_size) / (total_p + 1e-8)
        max_weight = (p_min * current_active_size + 1e-8) ** (-beta)

        for idx_val in idxs_list:
            p = self.sum_tree[idx_val] / (total_p + 1e-8)
            w = (p * current_active_size + 1e-8) ** (-beta)
            weights.append(w / (max_weight + 1e-8))

        weights_np = np.array(weights, dtype=np.float32)
        return idxs_np, weights_np

    def sample_expert_batch(self, expert_indices_list):
        """Samples ONLY from the provided expert indices."""
        if not expert_indices_list: return None, None
        num_to_sample = min(self.batch_size, len(expert_indices_list))
        # Sample indices *from the expert list*
        chosen_expert_indices = np.random.choice(expert_indices_list, num_to_sample, replace=(len(expert_indices_list) < num_to_sample))

        # Calculate weights (can just be 1.0 for pretraining, or use actual priorities if needed)
        # For simplicity in pretraining, let's use uniform weights among experts sampled
        # Or better, use weights=1.0 as IS correction isn't the goal here.
        weights_np = np.ones(num_to_sample, dtype=np.float32)

        return chosen_expert_indices, weights_np


    def update_priorities(self, idxs, prios):
        # Expects NumPy arrays or lists
        idxs_list = list(idxs) if not isinstance(idxs, list) else idxs
        prios_list = list(prios) if not isinstance(prios, list) else prios

        for i, p in zip(idxs_list, prios_list):
            if not 0 <= i < self.max_size: continue # Check against buffer size
            assert p > 0
            p_alpha = p ** self.alpha
            self.sum_tree[i] = p_alpha
            self.min_tree[i] = p_alpha
            self.max_prio = max(self.max_prio, p)

# ----------------------------------------------------------------------------
# Atari preprocessing (No changes)
# ----------------------------------------------------------------------------
class AtariPreprocessor:
    # ... (AtariPreprocessor code remains exactly the same) ...
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
# C51 network (Dueling Architecture)
# ----------------------------------------------------------------------------
class C51Network(nn.Module):
    # ... (Dueling C51Network code remains exactly the same as previous version) ...
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
        self.fc_hidden = nn.Sequential(nn.Linear(dim, 512), nn.ReLU())
        self.value_stream = nn.Linear(512, N_ATOMS)
        self.advantage_stream = nn.Linear(512, num_actions * N_ATOMS)

    def forward(self, x):
        features = self.feature(x); hidden = self.fc_hidden(features)
        value_logits = self.value_stream(hidden).view(-1, 1, N_ATOMS)
        advantage_logits = self.advantage_stream(hidden).view(-1, self.num_actions, N_ATOMS)
        mean_advantage_logits = advantage_logits.mean(dim=1, keepdim=True)
        q_logits = value_logits + (advantage_logits - mean_advantage_logits)
        return F.softmax(q_logits, dim=2)

# ----------------------------------------------------------------------------
# Agent (Implements pretraining and phased schedules)
# ----------------------------------------------------------------------------
class RainbowDQfDAgent:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.support = SUPPORT_CPU.to(self.device)

        self.env = gym.make(args.env_name, render_mode='rgb_array')
        self.env.action_space.seed(self.seed)

        # Store hyperparameters directly
        self.gamma = args.gamma
        self.n_step = args.n_step
        self.initial_beta = args.beta # Initial beta
        self.beta_anneal = args.beta_anneal_steps
        self.init_margin = args.supervised_margin
        self.margin_anneal = args.margin_anneal_steps
        self.lambda_sup = args.lambda_supervised
        self.target_update = args.target_update
        self.eval_epsilon = args.eval_epsilon
        self.max_episode_steps = args.max_episode_steps
        # Epsilon schedule parameters
        self.eps_start = args.eps_start
        self.eps_mid = 0.1 # Epsilon after phase 1
        self.eps_end = args.eps_end
        self.eps_phase1_steps = 50000 # Steps for 1.0 -> 0.1
        self.eps_phase2_steps = 200000 # Steps for 0.1 -> final_eps (total 250k decay)

        self.prep = AtariPreprocessor(args.frame_stack)
        self.frame_skip = args.frame_skip

        obs_shape = (args.frame_stack, 84, 84)
        # Use Non-Pinned Memory Buffer
        self.memory = PrioritizedReplayBuffer(obs_shape, args.memory_size,
                                              args.batch_size, alpha=args.alpha,
                                              n_step=self.n_step, gamma=self.gamma)

        self.expert_idxs = set()
        self.expert_indices_list = [] # Keep list for faster sampling during pretrain
        if args.expert_data_path and args.num_expert_transitions > 0:
            print(f"Loading expert data from {args.expert_data_path}...")
            try:
                if not os.path.exists(args.expert_data_path):
                     raise FileNotFoundError(f"Expert data file not found: {args.expert_data_path}")
                with open(args.expert_data_path, 'rb') as f: expert_data = pickle.load(f)
                num_to_load = min(len(expert_data), args.num_expert_transitions, self.memory.max_size)
                print(f"Prefilling buffer with {num_to_load} expert transitions...")
                for i in range(num_to_load):
                     s, a, r, ns, d = expert_data[i]
                     write_ptr = self.memory.ptr
                     # Directly write to numpy buffers
                     self.memory.obs_buf[write_ptr] = s
                     self.memory.next_obs_buf[write_ptr] = ns
                     self.memory.acts_buf[write_ptr] = a
                     self.memory.rews_buf[write_ptr] = r
                     self.memory.done_buf[write_ptr] = float(d)
                     prio = self.memory.max_prio ** self.memory.alpha
                     self.memory.sum_tree[write_ptr] = prio
                     self.memory.min_tree[write_ptr] = prio
                     self.expert_idxs.add(write_ptr)
                     self.expert_indices_list.append(write_ptr) # Store index
                     self.memory.ptr = (self.memory.ptr + 1) % self.memory.max_size
                     self.memory.size = min(self.memory.size + 1, self.memory.max_size)
                print(f"Buffer prefilled. Size: {self.memory.size}/{self.memory.max_size}")
            # ... (error handling remains same) ...
            except FileNotFoundError as e: print(f"Error: {e}. Warning: Continuing without expert data.")
            except Exception as e: print(f"Error loading expert data: {e}. Warning: Continuing without expert data.")
        else: print("Skipping expert data prefilling.")


        self.num_actions = self.env.action_space.n
        self.online = C51Network(self.num_actions, args.frame_stack).to(self.device)
        self.target = C51Network(self.num_actions, args.frame_stack).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=args.lr, eps=1.5e-4)

        # LR Schedule uses args.total_steps (should be set to 600k)
        def lr_lambda(step):
            if step < args.warmup_steps:
                return float(step) / float(max(1, args.warmup_steps))
            # Use args.total_steps for the denominator
            progress = float(step - args.warmup_steps) / float(max(1, args.total_steps - args.warmup_steps))
            progress = min(1.0, progress)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.wandb = wandb
        self.total_steps = 0
        self.scaler = None # Keep AMP disabled
        print("Mixed Precision Disabled.")

    # --- Parameter Annealing Functions ---
    def beta(self):
        # Linear anneal from initial beta to 1.0 over beta_anneal steps
        return min(1.0, self.initial_beta + self.total_steps / self.beta_anneal * (1.0 - self.initial_beta))

    def margin(self):
        # Linear anneal from init_margin to 0 over margin_anneal steps
        return max(0.0, self.init_margin * (1.0 - self.total_steps / self.margin_anneal))

    def epsilon(self):
        # Piecewise linear decay based on the plan
        if self.total_steps < self.eps_phase1_steps:
            # Phase 1: 1.0 -> 0.1
            return self.eps_start - (self.eps_start - self.eps_mid) * (self.total_steps / self.eps_phase1_steps)
        elif self.total_steps < self.eps_phase1_steps + self.eps_phase2_steps:
            # Phase 2: 0.1 -> eps_end
            steps_in_phase2 = self.total_steps - self.eps_phase1_steps
            return self.eps_mid - (self.eps_mid - self.eps_end) * (steps_in_phase2 / self.eps_phase2_steps)
        else:
            # Phase 3: Stay at eps_end
            return self.eps_end

    def select_action(self, state, eval_mode=False):
        current_eps = self.eval_epsilon if eval_mode else self.epsilon()
        if random.random() < current_eps:
            return self.env.action_space.sample()

        # Use torch.from_numpy as state is numpy array from preprocessor
        state_tensor = torch.from_numpy(state).to(self.device).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            probs = self.online(state_tensor)
            q = (probs * self.support).sum(2)
            return int(q.argmax(1).item())

    @torch.no_grad()
    def projection(self, nxt_p, rew, done):
        # ... (Fully Vectorized projection code remains exactly the same) ...
        B = rew.size(0)
        Tz = rew.unsqueeze(1) + (1 - done.unsqueeze(1)) * ((self.gamma ** self.n_step) * self.support.unsqueeze(0))
        Tz = Tz.clamp(V_MIN, V_MAX); b = (Tz - V_MIN) / DELTA_Z
        l = b.floor().long(); u = b.ceil().long()
        l_eq_u = (l == u)
        p_l_orig = nxt_p * (u.float() - b); p_u_orig = nxt_p * (b - l.float())
        p_l = torch.where(l_eq_u, nxt_p, p_l_orig); p_u = torch.where(l_eq_u, torch.zeros_like(p_u_orig), p_u_orig)
        offset = torch.arange(0, B * N_ATOMS, N_ATOMS, device=self.device).unsqueeze(1)
        l_clamped = l.clamp(0, N_ATOMS - 1); u_clamped = u.clamp(0, N_ATOMS - 1)
        flat_l_idx = (l_clamped + offset).view(-1); flat_u_idx = (u_clamped + offset).view(-1)
        flat_p_l = p_l.view(-1); flat_p_u = p_u.view(-1)
        m_flat = torch.zeros(B * N_ATOMS, device=self.device)
        m_flat.scatter_add_(0, flat_l_idx, flat_p_l); m_flat.scatter_add_(0, flat_u_idx, flat_p_u)
        return m_flat.view(B, N_ATOMS)

    def update_model(self, pretrain_phase=False):
        # Pretrain phase only calculates supervised loss if expert samples are present
        # Regular phase calculates full loss

        if not pretrain_phase and self.memory.size < self.args.batch_size:
             # Only check buffer size if not pretraining (pretraining samples experts)
             # Or maybe check self.args.replay_start_size? Plan mentioned 5k. Add later if needed.
             return None

        current_beta = self.beta()

        if pretrain_phase:
            # Sample only expert indices
            idxs_np, weights_np = self.memory.sample_expert_batch(self.expert_indices_list)
            if idxs_np is None: return None # No experts to sample
        else:
            # Sample normally from the buffer
            idxs_np, weights_np = self.memory.sample_batch(current_beta)
            if idxs_np is None: return None

        # Convert numpy indices/weights to CPU tensors
        idxs = torch.from_numpy(idxs_np).long()
        weights = torch.from_numpy(weights_np).float()
        weights_gpu = weights.to(self.device).unsqueeze(1) # No non_blocking

        # Get data from NumPy buffers, create tensors, transfer to GPU
        # Using list comprehension for indexing numpy array might be slightly cleaner
        # Or use the idxs_np directly
        obs = torch.tensor(self.memory.obs_buf[idxs_np], device=self.device).float() / 255.0
        nxt = torch.tensor(self.memory.next_obs_buf[idxs_np], device=self.device).float() / 255.0
        acts = torch.tensor(self.memory.acts_buf[idxs_np], device=self.device).long().unsqueeze(1)
        rews = torch.tensor(self.memory.rews_buf[idxs_np], device=self.device).float()
        dones = torch.tensor(self.memory.done_buf[idxs_np], device=self.device).float()

        # Calculate supervised loss (always needed)
        curr_p_sup = self.online(obs) # Need Q values for margin loss
        current_margin_sup = self.margin() if not pretrain_phase else self.init_margin # Use initial margin for pretrain? Or 0.8? Let's use 0.8 fixed.
        if pretrain_phase: current_margin_sup = 0.8 # Fixed margin for pretrain as per plan phase 0

        Q_sup = (curr_p_sup * self.support).sum(2)
        aexp_sup = acts.squeeze(1)
        Qexp_sup = Q_sup[torch.arange(len(idxs)), aexp_sup] # Use actual batch size len(idxs)
        mloss_sup = torch.clamp(Q_sup + current_margin_sup - Qexp_sup.unsqueeze(1), min=0)
        mloss_sup.scatter_(1, aexp_sup.unsqueeze(1), 0.0)

        # Mask (only expert samples contribute to supervised loss)
        # In pretrain phase, all samples *should* be expert, but mask ensures correctness
        mask = torch.tensor([1.0 if i in self.expert_idxs else 0.0 for i in idxs_np],
                            device=self.device, dtype=torch.float32)
        num_expert_samples = mask.sum()
        sup_loss = (mloss_sup.sum(1) * mask).sum() / num_expert_samples if num_expert_samples > 0 else torch.tensor(0.0, device=self.device)

        if pretrain_phase:
            loss = sup_loss # Only supervised loss for pretraining
            weighted_dist_loss = torch.tensor(0.0) # Placeholder
            dist_loss_per_item = torch.zeros(len(idxs), device='cpu') # No prio update needed really, but avoids error later
        else:
            # --- Regular RL Loss Calculation ---
            # Target distribution calculation (Double DQN)
            with torch.no_grad():
                online_next_dist = self.online(nxt).detach() # Detach here explicitly
                online_next_q = (online_next_dist * self.support).sum(2)
                nxt_a = online_next_q.argmax(1)
                # Target net calculation requires no grad anyway, but safety doesn't hurt
                target_next_dist_all = self.target(nxt)
                nxt_dist = target_next_dist_all[torch.arange(len(idxs)), nxt_a]
                tgt = self.projection(nxt_dist, rews, dones)

            # Current distribution and loss calculation
            curr_p = curr_p_sup # Can reuse forward pass from supervised calc
            curr_pa = curr_p[torch.arange(len(idxs)), acts.squeeze(1)]
            dist_loss_per_item = -(tgt * torch.log(curr_pa + 1e-8)).sum(1)
            weighted_dist_loss = (dist_loss_per_item * weights_gpu.squeeze(1)).mean()

            # Combine losses
            loss = weighted_dist_loss + self.lambda_sup * sup_loss
            # --- End Regular RL Loss ---

        # Optimization
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        if not pretrain_phase:
            # LR Scheduler Step only during RL phase
            self.lr_scheduler.step()

            # Update priorities only during RL phase (use CPU tensors)
            prios_cpu = (dist_loss_per_item.detach().cpu() + 1e-6)
            # Pass original numpy indices and numpy priorities
            self.memory.update_priorities(idxs_np, prios_cpu.numpy())

            # Target Network Sync only during RL phase
            if self.total_steps % self.target_update == 0:
                self.target.load_state_dict(self.online.state_dict())

        # Return losses for logging
        return loss.item(), weighted_dist_loss.item(), sup_loss.item()


    def pretrain(self, num_pretrain_steps):
        """Performs supervised pretraining on expert data."""
        if not self.expert_indices_list:
            print("No expert data loaded, skipping pretraining.")
            return

        print(f"--- Starting Supervised Pretraining for {num_pretrain_steps} steps ---")
        pretrain_loss_sum = 0
        start_time = time.time()
        for step in range(num_pretrain_steps):
            result = self.update_model(pretrain_phase=True)
            if result is not None:
                loss_val, _, sup_loss_val = result
                pretrain_loss_sum += sup_loss_val # Log only supervised part

            if (step + 1) % 1000 == 0:
                 avg_loss = pretrain_loss_sum / 1000 if step >= 999 else pretrain_loss_sum / (step + 1)
                 print(f"  Pretrain Step: {step+1}/{num_pretrain_steps}, Avg Sup Loss: {avg_loss:.4f}")
                 if step >= 999: pretrain_loss_sum = 0 # Reset sum

        # Copy weights to target network after pretraining
        self.target.load_state_dict(self.online.state_dict())
        duration = time.time() - start_time
        print(f"--- Finished Pretraining in {duration:.2f}s ---")


    def train(self, num_episodes, eval_interval):
        consecutive_eval_success = 0 # For early stopping
        for ep in range(1, num_episodes + 1):
            obs, _ = self.env.reset(seed=self.seed + ep)
            state = self.prep.reset(obs)
            done = False
            ep_r = 0; ep_steps = 0; ep_loss_sum = 0
            ep_dist_loss_sum = 0; ep_sup_loss_sum = 0; ep_updates = 0
            start_time = time.time()

            while not done:
                # --- Episode Step Limit ---
                if self.max_episode_steps > 0 and ep_steps >= self.max_episode_steps:
                    done = True # Force truncation flag
                if done: break

                # --- Agent-Environment Interaction ---
                # Only start training updates after replay_start_size? Plan mentioned 5k.
                # Let's add this check before calling update_model
                can_train = self.memory.size >= self.args.replay_start_size

                a = self.select_action(state) # Epsilon is calculated based on self.total_steps
                tot_r = 0; frame_done = False; last_obs = None
                for _ in range(self.frame_skip):
                    nxt, rew, ter, tr, _ = self.env.step(a)
                    tot_r += rew; frame_done = ter or tr; last_obs = nxt
                    if frame_done: break
                ns = self.prep.step(last_obs)
                self.memory.store(state, a, tot_r, ns, frame_done)
                state = ns; done = frame_done or done
                ep_r += tot_r; self.total_steps += 1; ep_steps += 1

                # --- Model Update ---
                # Original plan used train_frequency=4, gradient_steps=1
                # This means update every 4 *env steps*
                if can_train and self.total_steps % self.args.train_frequency == 0:
                     # Perform gradient_steps updates
                     for _ in range(self.args.gradient_steps):
                         update_result = self.update_model(pretrain_phase=False)
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

            log_data = { # ... (logging data remains same) ...
                'Train/Episode Reward': ep_r, 'Train/Episode Length': ep_steps,
                'Train/Episode Duration (s)': ep_duration, 'Train/Average Loss': avg_loss,
                'Train/Average Distributional Loss': avg_dist_loss, 'Train/Average Supervised Loss': avg_sup_loss,
                'Parameters/Epsilon': self.epsilon(), 'Parameters/PER Beta': self.beta(),
                'Parameters/Supervised Margin': self.margin(), 'Parameters/Learning Rate': self.optimizer.param_groups[0]['lr'],
                'Memory/Size': self.memory.size}
            self.wandb.log(log_data, step=self.total_steps)


            # --- Evaluation Phase ---
            if ep % eval_interval == 0:
                avg_eval_reward, std_eval_reward = self.evaluate(n_episodes=5) # Use evaluate method
                print(f"--- Eval Results @ Ep {ep}: Avg Reward={avg_eval_reward:.2f} +/- {std_eval_reward:.2f} (over 5 eps) ---", flush=True)
                self.wandb.log({'Eval/Average Reward': avg_eval_reward,
                                'Eval/Std Reward': std_eval_reward}, step=self.total_steps)

                # Early stopping logic
                if avg_eval_reward >= 19.0:
                    consecutive_eval_success += 1
                    print(f"  Met target score! Consecutive successes: {consecutive_eval_success}/3")
                else:
                    consecutive_eval_success = 0
                    print(f"  Did not meet target score. Resetting consecutive successes.")

                if consecutive_eval_success >= 3:
                    print(f"\nTarget score reached and maintained for 3 consecutive evaluations. Stopping early at step {self.total_steps}.")
                    break # Stop training

            # Check total steps limit
            if self.total_steps >= self.args.total_steps:
                 print(f"\nReached maximum total steps ({self.args.total_steps}). Stopping training.")
                 break


    def evaluate(self, n_episodes=5):
        """Runs evaluation episodes."""
        print(f"--- Evaluating (Steps: {self.total_steps}) ---", flush=True)
        eval_rewards = []
        with torch.no_grad():
            for eval_ep in range(n_episodes):
                eval_obs, _ = self.env.reset(seed=self.seed + self.total_steps + eval_ep) # Use different seed base
                eval_state = self.prep.reset(eval_obs)
                eval_done = False; eval_rsum = 0; eval_steps = 0
                while not eval_done:
                    if self.max_episode_steps > 0 and eval_steps >= self.max_episode_steps:
                        eval_done = True
                    if eval_done: break

                    eval_a = self.select_action(eval_state, eval_mode=True) # Use fixed eval epsilon
                    eval_tot_r = 0; eval_frame_done = False; eval_last_obs = None
                    for _ in range(self.frame_skip):
                        eval_nxt, eval_rew, eval_ter, eval_tr, _ = self.env.step(eval_a)
                        eval_tot_r += eval_rew; eval_frame_done = eval_ter or eval_tr; eval_last_obs = eval_nxt
                        if eval_frame_done: break
                    eval_ns = self.prep.step(eval_last_obs)
                    eval_state = eval_ns; eval_rsum += eval_tot_r
                    eval_done = eval_frame_done or eval_done; eval_steps += 1
                eval_rewards.append(eval_rsum)
        return np.mean(eval_rewards), np.std(eval_rewards)


if __name__=='__main__':
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Environment (using plan defaults)
    p.add_argument('--env_name',default='ALE/Pong-v5')
    p.add_argument('--seed',type=int,default=777)
    p.add_argument('--frame_stack',type=int,default=4)
    p.add_argument('--frame_skip',type=int,default=4)
    p.add_argument('--max_episode_steps', type=int, default=27000)

    # Training Loop (using plan defaults focused on 600k goal)
    p.add_argument('--num_episodes',type=int,default=10000, help="Max episodes (effective limit is total_steps)") # Set high, let total_steps limit
    p.add_argument('--total_steps',type=int,default=600000, help="Total training steps for LR schedule & overall budget") # TARGET
    p.add_argument('--replay_start_size', type=int, default=5000, help="Steps before starting training updates") # From plan text
    p.add_argument('--train_frequency', type=int, default=4, help="Env steps per online net update") # From plan text
    p.add_argument('--gradient_steps', type=int, default=1, help="Gradient steps per online net update") # From plan text

    # Replay Buffer (using plan defaults)
    p.add_argument('--memory_size',type=int,default=100000)
    p.add_argument('--batch_size',type=int,default=64, help="Batch size (smaller as per plan)") # Plan Value
    p.add_argument('--n_step',type=int,default=5, help="N-step returns (plan value)") # Plan Value

    # PER (using plan defaults)
    p.add_argument('--alpha',type=float,default=0.5)
    p.add_argument('--beta',type=float,default=0.4, help="Initial PER beta")
    p.add_argument('--beta_anneal_steps',type=int,default=600000, help="Steps to anneal beta to 1.0 (match total_steps)") # Match total_steps

    # DQfD / Supervised Loss (using plan defaults)
    p.add_argument('--expert_data_path',default='./expert_data_pong.pkl')
    p.add_argument('--num_expert_transitions',type=int,default=20000)
    p.add_argument('--supervised_margin',type=float,default=0.8, help="Initial supervised loss margin")
    p.add_argument('--margin_anneal_steps',type=int,default=200000, help="Steps to anneal margin to 0") # Plan Value
    p.add_argument('--lambda_supervised',type=float,default=1.0)
    p.add_argument('--pretrain_steps', type=int, default=10000, help="Number of supervised pretraining steps") # From plan text

    # DQN Algorithm Details (using plan defaults)
    p.add_argument('--gamma',type=float,default=0.99)
    p.add_argument('--lr',type=float,default=6.25e-5, help="Learning rate for Adam") # Plan Value
    p.add_argument('--warmup_steps',type=int,default=10000, help="Number of steps for LR warmup") # Plan Value
    p.add_argument('--target_update',type=int,default=8000, help="Frequency (in steps) to update target network") # Plan Value
    p.add_argument('--eps_start',type=float,default=1.0)
    p.add_argument('--eps_end',type=float,default=0.01, help="Final epsilon after full decay") # Plan Value
    # Note: eps_decay argument is removed, decay is handled by fixed schedule in agent.epsilon()

    # Evaluation (using plan defaults)
    p.add_argument('--eval_interval',type=int,default=50)
    p.add_argument('--eval_epsilon', type=float, default=0.001)

    # Logging
    p.add_argument('--wandb_project',default='Rainbow-DQfD-Pong', help="WandB project name")
    p.add_argument('--wandb_run_name',default=None, help="WandB run name (defaults to auto-generated)")

    args=p.parse_args()

    # --- Final Hyperparameter Checks / Adjustments ---
    # Ensure total_steps used for LR schedule matches budget
    # (It does, set to 600k in args)
    # Ensure beta annealing matches total_steps
    args.beta_anneal_steps = args.total_steps

    if args.wandb_run_name is None:
        env_short_name = args.env_name.split('/')[-1].replace('-v5', '')
        # Include key params in run name for easier WandB tracking
        args.wandb_run_name = f"{env_short_name}-DQfD_B{args.batch_size}_N{args.n_step}_LR{args.lr:.1e}-T{int(time.time())}"

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    agent=RainbowDQfDAgent(args)

    # --- Execute Pretraining ---
    agent.pretrain(args.pretrain_steps)

    # --- Execute Main Training ---
    agent.train(args.num_episodes, args.eval_interval)

    wandb.finish()
    print("Training finished.")