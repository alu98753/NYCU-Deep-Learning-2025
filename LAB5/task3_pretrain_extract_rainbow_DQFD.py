# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL / DQFD Integration
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from copy import deepcopy 

gym.register_envs(ale_py)
torch.set_num_threads(4)
torch.set_num_interop_threads(4)  
cv2.setNumThreads(0)  
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if not isinstance(obs, np.ndarray): obs = np.array(obs)
        if len(obs.shape) == 3 and obs.shape[2] == 3: gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif len(obs.shape) == 2: gray = obs
        elif len(obs.shape) == 3 and obs.shape[2] == 1: gray = obs.squeeze(axis=2)
        else: raise ValueError(f"Unexpected obs shape: {obs.shape}")
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

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
        if parent != 0: self._propagate(parent, change)
    def _retrieve(self, idx, s):  
        left = 2 * idx + 1; right = left + 1
        if left >= len(self.tree): return idx
        return self._retrieve(left, s) if s <= self.tree[left] else self._retrieve(right, s - self.tree[left])
    def total(self): return self.tree[0]  
    def add(self, p, data):  
        idx = self.write + self.capacity - 1
        self.data[self.write] = data; self.update(idx, p)
        self.write += 1; self.write %= self.capacity
        if self.n_entries < self.capacity: self.n_entries += 1
    def update(self, idx, p):  
        change = p - self.tree[idx]; self.tree[idx] = p
        if idx != 0: self._propagate(idx, change)
    def get(self, s): 
        idx = self._retrieve(0, s); dataIdx = idx - self.capacity + 1
        dataIdx = idx - self.capacity + 1

        if not (0 <= dataIdx < self.n_entries):
            return None, None, None
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.5, beta=0.4, beta_increment_per_sampling=1e-5, epsilon=1e-6, expert_epsilon=1.0): # Default expert_epsilon=1.0
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon # Epsilon for agent transitions
        self.expert_epsilon = expert_epsilon # Epsilon for expert transitions
        self.max_priority = 1.0
        self.n_entries = 0
        print(f"PER Initialized: alpha={alpha}, beta0={beta}, agent_eps={epsilon}, expert_eps={expert_epsilon}")

    def __len__(self):
        return self.tree.n_entries

    def add(self, transition, is_expert=False):
        priority = self.max_priority 
        self.tree.add(priority, transition + (is_expert,))

    def sample(self, batch_size):
        batch = []; idxs = []; priorities = []; is_experts = []
        segment = self.tree.total() / batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        current_size = len(self)

        if current_size < batch_size or self.tree.total() == 0:
            # print(f"Warning: Buffer size ({current_size}) or total priority ({self.tree.total()}) insufficient for batch size ({batch_size}). Cannot sample.")
            return None, None, None, None, None, None, None, None, None

        sampled_count = 0
        attempts = 0 # 防止無限循環
        max_attempts = batch_size * 5 

        while sampled_count < batch_size and attempts < max_attempts:
            attempts += 1
            a = segment * (sampled_count % batch_size) 
            b = segment * ((sampled_count % batch_size) + 1)
            s = random.uniform(a, b); s = min(s, self.tree.total() - 1e-7)

            try:
                get_result = self.tree.get(s) 
                if get_result is None or get_result[0] is None:
                    # print(f"Debug: tree.get returned None for s={s}. Retrying sample.")
                    continue # get 失敗，跳過此次嘗試，進行下一次採樣

                idx, priority, data = get_result

                if data is None or not isinstance(data, tuple) or len(data) != 6:
                    print(f"Warning: Skipping sample due to invalid data format: {data}")
                    continue # data 格式錯誤，跳過此次嘗試

                # 如果一切正常，添加數據
                priorities.append(priority); idxs.append(idx); batch.append(data[:-1])
                is_experts.append(data[-1])
                sampled_count += 1 # 成功採樣

            except Exception as e:
                print(f"Error during PER sampling: s={s}, total={self.tree.total()}, segment=[{a},{b}]"); raise e

        # --- 處理未能採集滿 batch 的情況 ---
        actual_batch_size = len(batch)
        if actual_batch_size == 0:
             print("Warning: Failed to sample any valid transitions.")
             return None, None, None, None, None, None, None, None, None
        if actual_batch_size < batch_size:
             print(f"Warning: Sampled only {actual_batch_size}/{batch_size} transitions after {attempts} attempts.")
        # ------------------------------------

        # Importance sampling weights calculation (使用 actual_batch_size?)
        # 權重計算應該基於成功採樣的樣本
        sampling_probabilities = np.array(priorities) / self.tree.total()
        weights = np.power(self.tree.n_entries * (sampling_probabilities + 1e-10), -self.beta)
        weights /= (weights.max() + 1e-10) # 根據當前最大權重歸一化

        states, actions, rewards, next_states, dones = zip(*batch)
        states_np=np.array(states); next_states_np=np.array(next_states)
        actions_tensor=torch.tensor(actions,dtype=torch.int64); rewards_tensor=torch.tensor(rewards,dtype=torch.float32); dones_tensor=torch.tensor(dones,dtype=torch.float32); weights_tensor=torch.tensor(weights,dtype=torch.float32); is_experts_tensor=torch.tensor(is_experts,dtype=torch.bool)

        num_experts_in_batch=is_experts_tensor.sum().item(); proportion_experts=num_experts_in_batch/actual_batch_size

        return states_np, actions_tensor, rewards_tensor, next_states_np, dones_tensor, is_experts_tensor, weights_tensor, idxs, proportion_experts


    def update_priorities(self, indices, errors, is_expert_flags): # is_expert_flags is numpy array
        # --- <<< Using standard PER formula p = (|error| + epsilon)^alpha >>> ---
        # --- <<< Epsilon value depends on whether it's an expert transition >>> ---
        if not isinstance(is_expert_flags, np.ndarray):
             is_expert_flags = np.array(is_expert_flags) # Ensure it's numpy array
        if not isinstance(errors, np.ndarray):
             errors = np.array(errors) # Ensure errors is numpy array

        # Select epsilon based on the flag
        epsilons_to_add = np.where(is_expert_flags, self.expert_epsilon, self.epsilon)

        # Calculate priorities: (|error| + selected_epsilon) ^ alpha
        # Ensure errors are non-negative before adding epsilon
        priorities = (np.abs(errors) + epsilons_to_add) ** self.alpha
        # --------------------------------------------------------------------

        # Clip priorities to avoid extreme values if necessary (optional)
        # priorities = np.clip(priorities, 1e-6, None)

        for i, idx in enumerate(indices):
            # Ensure index is valid before updating
            if idx < self.capacity -1 + self.n_entries and idx >= self.capacity -1 : # check bounds of tree leaf nodes
                 self.tree.update(idx, priorities[i])
            # else: # This case indicates an invalid index from sampling, should be rare
            #    print(f"Warning: Invalid index {idx} provided to update_priorities. Skipping.")


        if priorities.size > 0:
            self.max_priority = max(self.max_priority, priorities.max())

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None: nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0; nn.init.uniform_(m.bias, -bound, bound)


class DuelingC51DQN(nn.Module): 
    def __init__(self, num_actions, num_atoms=51, vmin=-10, vmax=10):
        super(DuelingC51DQN, self).__init__()
        self.num_actions = num_actions; self.num_atoms = num_atoms; self.vmin = vmin; self.vmax = vmax
        self.support = torch.linspace(vmin, vmax, num_atoms); self.delta_z = (vmax - vmin) / (num_atoms - 1)
        self.conv_base = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten())
        self.flattened_size = 64 * 7 * 7
        self.value_stream = nn.Sequential(nn.Linear(self.flattened_size, 512), nn.ReLU(), nn.Linear(512, self.num_atoms))
        self.advantage_stream = nn.Sequential(nn.Linear(self.flattened_size, 512), nn.ReLU(), nn.Linear(512, self.num_actions * self.num_atoms))
        self.register_buffer("support_buf", self.support); self.register_buffer("delta_z_buf", torch.tensor(self.delta_z))
    def forward(self, x): 
        x = x / 255.0; features = self.conv_base(x)
        value_logits = self.value_stream(features); advantage_logits = self.advantage_stream(features)
        value_logits = value_logits.view(-1, 1, self.num_atoms); advantage_logits = advantage_logits.view(-1, self.num_actions, self.num_atoms)
        mean_advantage_logits = advantage_logits.mean(1, keepdim=True)
        q_logits = value_logits + advantage_logits - mean_advantage_logits; return q_logits
    def get_expected_q_values(self, x): 
        q_logits = self.forward(x); q_probs = F.softmax(q_logits, dim=2)
        expected_q = torch.sum(q_probs * self.support_buf, dim=2); return expected_q


class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.args = args
        # Env setup (unchanged)
        if not getattr(args, 'generate_expert_data', False): self.env = gym.make(env_name, frameskip=1)
        else: self.env = None
        self.test_env = gym.make(env_name, frameskip=1, render_mode=None)
        self.num_actions = self.test_env.action_space.n

        self.preprocessor = AtariPreprocessor(frame_stack=args.frame_stack)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Network setup (unchanged)
        self.q_net = DuelingC51DQN(self.num_actions, args.num_atoms, args.vmin, args.vmax).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DuelingC51DQN(self.num_actions, args.num_atoms, args.vmin, args.vmax).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict()); self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=1.5e-4 if 'atari' in args.env_name.lower() else 1e-8)
        self.support = self.q_net.support_buf.to(self.device); self.delta_z = self.q_net.delta_z_buf.to(self.device)
        self.num_atoms = args.num_atoms; self.vmin = args.vmin; self.vmax = args.vmax

        # Parameters (unchanged)
        self.batch_size = args.batch_size; self.gamma = args.discount_factor
        self.epsilon_start = args.epsilon_start; self.epsilon_decay_steps = args.epsilon_decay_steps
        self.epsilon_min = args.epsilon_min
        self.env_count = 0; self.decision_steps = 0; self.train_count = 0; self.best_reward = -21.0
        self.max_episode_steps = args.max_episode_steps; self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_decision_step = args.train_per_decision_step
        self.frame_skip = args.frame_skip

        self.val_states = None
        self.val_actions = None

        if not getattr(args, 'generate_expert_data', False): # ... (save_dir, memory, n_step setup) ...
            self.save_dir = os.path.join(args.save_dir, f"{args.wandb_run_name}_fs{self.frame_skip}_{time.strftime('%Y%m%d-%H%M%S')}", f"{env_name.replace('/', '_')}"); os.makedirs(self.save_dir, exist_ok=True)
            beta_inc=(1.0-args.per_beta0)/args.per_beta_steps if args.per_beta_steps>0 else 0.0
            self.memory=PrioritizedReplayBuffer(args.memory_size, alpha=args.per_alpha, beta=args.per_beta0, beta_increment_per_sampling=beta_inc, epsilon=args.per_epsilon, expert_epsilon=args.expert_epsilon)
            self.n_step=args.n_step
            self.n_step_buffer=deque(maxlen=self.n_step)
            
            if args.load_expert_data: 
                self.load_expert_data(args.load_expert_data, args.val_split_ratio)
        else: # ... (gen mode setup) ...
            self.save_dir=None;self.memory=None
            self.n_step_buffer=None;self.n_step=1
            self.val_states=None;self.val_actions=None
            
    def load_expert_data(self, expert_data_path, val_split_ratio):
        if not os.path.exists(expert_data_path):
            print(f"Warning: Expert data path not found: {expert_data_path}"); return
        if not (0 < val_split_ratio < 1):
             print("Warning: Invalid val_split_ratio. Skipping expert data loading / validation split.")
             return

        try:
            print(f"Loading expert data from {expert_data_path}...")
            expert_data = np.load(expert_data_path)
            states = expert_data['states']
            actions = expert_data['actions']
            rewards = expert_data['rewards']
            next_states = expert_data['next_states']
            dones = expert_data['dones']
            num_total = len(states)
            print(f"Total expert transitions found: {num_total}")

            # Shuffle indices and split
            indices = np.arange(num_total)
            np.random.shuffle(indices)
            split_idx = int(num_total * (1 - val_split_ratio))
            train_indices = indices[:split_idx]
            val_indices = indices[split_idx:]

            num_train = len(train_indices)
            num_val = len(val_indices)
            print(f"Splitting into {num_train} training samples and {num_val} validation samples.")

            # Add training data to replay buffer
            num_added = 0
            for i in train_indices:
                transition = (states[i], actions[i], rewards[i], next_states[i], dones[i])
                self.memory.add(transition, is_expert=True)
                num_added += 1
            print(f"Added {num_added} expert training transitions to replay buffer.")
            print(f"Replay buffer size after loading: {len(self.memory)}")

            # Store validation data separately as tensors on device
            if num_val > 0:
                self.val_states = torch.from_numpy(states[val_indices].astype(np.float32)).to(self.device)
                self.val_actions = torch.from_numpy(actions[val_indices]).long().to(self.device) # Ensure long for gather
                print(f"Stored {num_val} validation transitions internally.")
            else:
                print("Warning: No validation samples created due to split ratio or data size.")
                self.val_states = None
                self.val_actions = None

        except Exception as e:
            print(f"Error loading or splitting expert data from {expert_data_path}: {e}")
            self.val_states = None # Ensure val data is None on error
            self.val_actions = None

    def _calculate_val_loss(self):
        if self.val_states is None or len(self.val_states) == 0:
            return float('inf') # Return infinity if no validation data

        self.q_net.eval() # Set model to evaluation mode
        total_val_loss_je = 0.0
        num_val_batches = 0

        with torch.no_grad():
            # Process validation data in batches to avoid OOM
            for i in range(0, len(self.val_states), self.batch_size):
                batch_states = self.val_states[i : i + self.batch_size]
                batch_actions = self.val_actions[i : i + self.batch_size]

                if len(batch_states) == 0: continue # Skip empty batch

                # --- Calculate Supervised Loss (J_E) on validation batch ---
                q_logits_val = self.q_net(batch_states)
                q_exp_val = (F.softmax(q_logits_val, dim=2) * self.support).sum(2)
                q_exp_val_action = q_exp_val.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                action_mask = torch.ones_like(q_exp_val).bool()
                action_mask.scatter_(1, batch_actions.unsqueeze(1), False)
                q_exp_masked = torch.where(action_mask, q_exp_val, torch.tensor(-float('inf'), device=self.device))
                max_non_expert_q = q_exp_masked.max(1).values
                loss_je_elementwise = F.relu(max_non_expert_q + self.args.margin - q_exp_val_action)
                # ----------------------------------------------------------

                total_val_loss_je += loss_je_elementwise.mean().item() # Average loss for the batch
                num_val_batches += 1

        self.q_net.train() # Set model back to training mode

        return total_val_loss_je / num_val_batches if num_val_batches > 0 else float('inf')

    def _evaluate_on_val_set(self):
        if self.val_states is None or len(self.val_states) == 0:
            return float('inf'), 0.0 # Return default values

        self.q_net.eval()
        total_val_loss_je = 0.0
        total_correct_matches = 0 # <<< Initialize counter
        total_samples = 0

        with torch.no_grad():
            for i in range(0, len(self.val_states), self.batch_size):
                batch_states = self.val_states[i : i + self.batch_size]
                batch_actions = self.val_actions[i : i + self.batch_size]

                if len(batch_states) == 0: continue

                # Calculate J_E Loss (as before)
                q_logits_val = self.q_net(batch_states)
                q_exp_val = (F.softmax(q_logits_val, dim=2) * self.support).sum(2)
                q_exp_val_action = q_exp_val.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
                # ... (rest of J_E calculation) ...
                action_mask=torch.ones_like(q_exp_val).bool(); action_mask.scatter_(1,batch_actions.unsqueeze(1),False); q_exp_masked=torch.where(action_mask,q_exp_val,torch.tensor(-float('inf'),device=self.device)); max_non_expert_q=q_exp_masked.max(1).values; loss_je_elementwise=F.relu(max_non_expert_q+self.args.margin-q_exp_val_action);
                total_val_loss_je += loss_je_elementwise.sum().item()

                # <<< --- Calculate Action Matching Accuracy --- >>>
                predicted_actions = q_exp_val.argmax(dim=1) # Get greedy actions
                correct_matches = (predicted_actions == batch_actions).sum().item()
                total_correct_matches += correct_matches
                # <<< -------------------------------------- >>>
                total_samples += len(batch_actions)

        self.q_net.train()

        avg_val_loss = total_val_loss_je / total_samples if total_samples > 0 else float('inf')
        avg_accuracy = total_correct_matches / total_samples if total_samples > 0 else 0.0

        return avg_val_loss, avg_accuracy


    def pretrain(self):
        if not self.args.load_expert_data or self.args.pretrain_steps <= 0 or self.memory is None or len(self.memory) == 0:
             print("Skipping pre-training..."); self.target_net.load_state_dict(self.q_net.state_dict()); return

        # Handle separate pre-train LR (code from before)
        pretrain_optimizer = None; using_separate_pretrain_lr = False
        if self.args.pretrain_lr is not None and abs(self.args.pretrain_lr - self.args.lr) > 1e-9:
             print(f"Using separate pre-training LR: {self.args.pretrain_lr}"); pretrain_optimizer = optim.Adam(self.q_net.parameters(), lr=self.args.pretrain_lr, eps=1.5e-4 if 'atari' in self.args.env_name.lower() else 1e-8); using_separate_pretrain_lr = True
        else: print(f"Using main LR ({self.args.lr}) for pre-training."); pretrain_optimizer = self.optimizer

        # Determine if early stopping is active and get threshold
        use_early_stopping = self.val_states is not None and self.args.patience > 0 and self.args.val_split_ratio > 0
        accuracy_threshold = self.args.accuracy_threshold # Get threshold from args

        if use_early_stopping: print(f"Starting pre-training for max {self.args.pretrain_steps} steps with validation-based early stopping..."); print(f"(Validate every {self.args.validate_every} steps, patience={self.args.patience} after Acc >= {accuracy_threshold:.1%}, min_delta={self.args.min_delta})");
        else: print(f"Starting pre-training for fixed {self.args.pretrain_steps} steps...")

        self.q_net.train()
        best_val_loss = float('inf'); patience_counter = 0; best_model_state_dict = None
        start_time = time.time(); last_logged_val_loss = float('nan'); last_logged_val_acc = float('nan')
        accuracy_threshold_met = False # Flag

        for step in range(self.args.pretrain_steps):
            # --- Training Step on Expert Data ---
            # ... (Sampling, Filtering, Loss Calc, Optimization - as before) ...
            sample_result=self.memory.sample(self.batch_size);
            if sample_result is None: continue
            states_np,actions,_,_,_,is_experts,weights,indices,_=sample_result; expert_mask_np=is_experts.cpu().numpy(); expert_indices_np=np.where(expert_mask_np)[0];
            if len(expert_indices_np)==0: continue;
            expert_states_np=states_np[expert_indices_np]; expert_actions=actions[expert_indices_np].to(self.device); expert_weights=weights[expert_indices_np].to(self.device); expert_indices_memory=[indices[i] for i in expert_indices_np]; expert_is_expert_flags=expert_mask_np[expert_indices_np]; expert_states=torch.from_numpy(expert_states_np.astype(np.float32)).to(self.device);
            q_logits_expert=self.q_net(expert_states); q_exp_expert=(F.softmax(q_logits_expert,dim=2)*self.support).sum(2); q_exp_expert_action=q_exp_expert.gather(1,expert_actions.unsqueeze(1)).squeeze(1); action_mask=torch.ones_like(q_exp_expert).bool(); action_mask.scatter_(1,expert_actions.unsqueeze(1),False); q_exp_masked=torch.where(action_mask,q_exp_expert,torch.tensor(-float('inf'),device=self.device)); max_non_expert_q=q_exp_masked.max(1).values; loss_je_elementwise=F.relu(max_non_expert_q+self.args.margin-q_exp_expert_action); loss_je=(expert_weights*loss_je_elementwise).mean();
            loss_l2=torch.tensor(0.0,device=self.device);
            if self.args.lambda2>0:
                for param in self.q_net.parameters():
                     if param.ndim>1: loss_l2+=torch.sum(param**2)
            total_loss=self.args.lambda1*loss_je + self.args.lambda2*loss_l2
            pretrain_optimizer.zero_grad(); total_loss.backward(); torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.args.grad_clip); pretrain_optimizer.step() # Use pretrain_optimizer
            priorities_update=loss_je_elementwise.abs().detach().cpu().numpy(); self.memory.update_priorities(expert_indices_memory, priorities_update, expert_is_expert_flags)
            # ----------------------------------

            # --- Validation Check ---
            if use_early_stopping and (step + 1) % self.args.validate_every == 0:
                current_val_loss, current_val_acc = self._evaluate_on_val_set()
                elapsed_time = time.time() - start_time
                last_logged_val_loss = current_val_loss; last_logged_val_acc = current_val_acc

                print(f"Pre-train Step [{step+1}/{self.args.pretrain_steps}], Val Loss JE: {current_val_loss:.4f}, Val Acc: {current_val_acc:.4f}, Time: {elapsed_time:.1f}s")

                # --- <<< Early Stopping Logic with Accuracy Threshold >>> ---
                # Update best loss regardless of accuracy, but only check patience if accuracy met
                if current_val_loss < best_val_loss - self.args.min_delta:
                     best_val_loss = current_val_loss
                     best_model_state_dict = deepcopy(self.q_net.state_dict())
                     print(f"  -> Validation loss improved to {best_val_loss:.4f}. Best state saved.")
                     # Reset patience if loss improves WHILE accuracy threshold was already met
                     if accuracy_threshold_met:
                          patience_counter = 0
                # Check if accuracy threshold is met NOW
                if current_val_acc >= accuracy_threshold:
                    if not accuracy_threshold_met:
                        print(f"  -> Validation accuracy threshold ({accuracy_threshold:.1%}) MET! Patience check active.")
                        accuracy_threshold_met = True
                        patience_counter = 0 # Reset patience when threshold is first met

                    # If accuracy met, check if loss has stopped improving (patience check)
                    # This part only runs if accuracy_threshold_met is True
                    if not (current_val_loss < best_val_loss - self.args.min_delta):
                         patience_counter += 1
                         print(f"  -> Validation loss did not improve enough for {patience_counter}/{self.args.patience} checks (Acc OK).")
                    # else: # Loss improved, patience already reset above when best_val_loss updated
                    #    pass

                    # Check patience ONLY if accuracy threshold was met
                    if patience_counter >= self.args.patience:
                        print(f"--- Early stopping pre-training at step {step+1} (Loss plateaued after Acc >= {accuracy_threshold:.1%}) ---")
                        break # Exit pre-training loop
                else:
                     # Accuracy threshold not yet met
                     print(f"  -> Validation accuracy ({current_val_acc:.4f}) below threshold ({accuracy_threshold:.1%}). Patience check deferred.")
                     # Reset the loss patience counter if accuracy drops below threshold? Or keep it?
                     # Let's reset it - we want sustained high accuracy AND loss plateau.
                     patience_counter = 0
                     if accuracy_threshold_met: # If it *was* met before and dropped
                         print("  -> Accuracy dropped below threshold. Resetting loss patience.")
                         accuracy_threshold_met = False # Reset flag
                # --- <<< End Early Stopping Logic >>> ---

            # --- Logging (every 1000 steps, includes validation results if calculated) ---
            if (step + 1) % 1000 == 0:
                log_data = {
                    # <<< Log the step value for the custom x-axis >>>
                    "Pretrain/Step": step + 1,
                    # --- Rest of the metrics ---
                    "Pretrain/JE Loss (Batch)": loss_je.item(),
                    "Pretrain/L2 Loss": loss_l2.item()
                }
                if not np.isnan(last_logged_val_loss): log_data["Pretrain/Validation JE Loss"] = last_logged_val_loss
                if not np.isnan(last_logged_val_acc): log_data["Pretrain/Validation Accuracy"] = last_logged_val_acc

                if wandb.run is not None:
                    # <<< REMOVE step= argument here >>>
                    wandb.log(log_data)
                # -------------------------------------------

                # Also print training loss if not validating this step
                if not (use_early_stopping and (step + 1) % self.args.validate_every == 0):
                    print(f"Pre-train Step [{step+1}/{self.args.pretrain_steps}], Loss JE (batch): {loss_je.item():.4f}, L2 Loss: {loss_l2.item():.4f}")
        
        # --- Post Pre-training ---
        # <<< -------------------------------- >>>
        # ... (Load best weights if found, update target net - unchanged) ...
        if use_early_stopping and best_model_state_dict is not None: print(f"Loading best model weights from pre-training (Val Loss: {best_val_loss:.4f})."); self.q_net.load_state_dict(best_model_state_dict)
        elif not use_early_stopping: print(f"Finished fixed {self.args.pretrain_steps} pre-training steps.")
        else: print(f"Finished pre-training (max steps reached or no validation improvement). Using weights from final step.")
        self.target_net.load_state_dict(self.q_net.state_dict()); print("Pre-training phase complete. Target network updated.")

    def _get_current_epsilon(self):
        fraction = min(1.0, self.decision_steps / self.epsilon_decay_steps); current_epsilon = self.epsilon_start + fraction * (self.epsilon_min - self.epsilon_start); return max(self.epsilon_min, current_epsilon)

    def select_action(self, state, use_greedy=False): 
        current_epsilon = 0.0 if use_greedy else self._get_current_epsilon();
        if random.random() < current_epsilon: return self.test_env.action_space.sample()
        state_np = np.array(state, dtype=np.float32); state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        with torch.no_grad(): expected_q = self.q_net.get_expected_q_values(state_tensor); action = expected_q.argmax().item()
        return action

    def _calculate_n_step_info(self): 
        R_n_step = 0.0; S_t, A_t = self.n_step_buffer[0][:2]; current_gamma = 1.0; true_n = len(self.n_step_buffer)
        for i in range(true_n): 
            s, a, r_accum_raw, ns, d = self.n_step_buffer[i]
            R_n_step += current_gamma * r_accum_raw
            current_gamma *= (self.gamma ** self.frame_skip)
            if d:
                true_n = i + 1
                break
        S_tpn = self.n_step_buffer[true_n - 1][3]; D_tpn = self.n_step_buffer[true_n - 1][4]; return S_t, A_t, R_n_step, S_tpn, D_tpn

    def run(self, episodes):
        if getattr(self.args, 'generate_expert_data', False): print("Skipping training run (generation mode)."); return

        # <<< Call pretrain before starting the main loop >>>
        self.pretrain()
        print("Starting training run...")

        start_time = time.time()
        for ep in range(episodes): # ... (reset env, state, buffers - unchanged) ...
            obs, _ = self.env.reset(); state = self.preprocessor.reset(obs)
            if "Pong" in self.args.env_name: fire_obs, _, _, _, _ = self.env.step(1); state = self.preprocessor.step(fire_obs)
            done = False; total_raw_reward_episode = 0.0; decision_step_count = 0; self.n_step_buffer.clear()

            while not done:
                if self.max_episode_steps > 0 and decision_step_count >= self.max_episode_steps: break
                current_decision_step_state = state
                action = self.select_action(current_decision_step_state, use_greedy=False)

                # Frame Skip (unchanged - accumulates raw reward)
                accumulated_raw_reward = 0.0; frame_done = False; last_obs = None
                for fs_step in range(self.frame_skip):
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    self.env_count += 1; accumulated_raw_reward += reward
                    frame_done = terminated or truncated; last_obs = next_obs
                    if frame_done: break
                if last_obs is None: print(f"Warning: last_obs is None."); break
                next_state = self.preprocessor.step(last_obs)

                # N-step Buffer (unchanged - stores raw reward)
                self.n_step_buffer.append((current_decision_step_state, action, accumulated_raw_reward, next_state, frame_done))

                if len(self.n_step_buffer) >= self.n_step:
                    S_t, A_t, R_n_step_raw, S_tpn, D_tpn = self._calculate_n_step_info()
                    # <<< Add AGENT transition to buffer >>>
                    transition_data = (S_t, A_t, R_n_step_raw, S_tpn, D_tpn)
                    self.memory.add(transition_data, is_expert=False) # Set is_expert=False

                state = next_state; done = frame_done
                total_raw_reward_episode += accumulated_raw_reward
                decision_step_count += 1; self.decision_steps += 1

                # Training Step (unchanged - calls self.train)
                if self.decision_steps >= self.replay_start_size:
                    for _ in range(self.train_per_decision_step): self.train()

                # Saving model based on decision steps (unchanged)
                if self.decision_steps > 0 and self.decision_steps % 200000 == 0 and self.decision_steps <= 1000000:
                    save_filename = f"LAB5_313554044_task3_pong{self.decision_steps}.pt"
                    save_path = os.path.join(self.save_dir, save_filename)
                    torch.save(self.q_net.state_dict(), save_path)
                    print(f"Saved intermediate model at step {self.decision_steps}: {save_path}")
                # === End Saving ===

            # End of Episode Logging & Eval & Saving (unchanged)
            # ... (logging, evaluate call, best model save) ...
            # (Inside logging: ensure memory.beta access is safe if memory is None)
            elapsed_time = time.time() - start_time
            steps_per_sec = self.decision_steps / elapsed_time if elapsed_time > 0 else 0
            mem_beta = self.memory.beta if self.memory else 0
            mem_len = len(self.memory) if self.memory else 0
            print(f"[End Ep {ep}] Raw Reward: {total_raw_reward_episode:.2f}, DecSteps: {decision_step_count}, TotalDecSteps: {self.decision_steps}, EnvSteps: {self.env_count}, TrainCount: {self.train_count}, Eps: {self._get_current_epsilon():.4f}, Mem: {mem_len}, Beta: {mem_beta:.3f}, SPS(Dec): {steps_per_sec:.0f}")
            if not getattr(self.args, 'generate_expert_data', False):
                log_dict = { # ... (create log_dict) ...
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
                if ep % self.args.eval_interval == 0: # ... (evaluation logic) ...
                    eval_raw_reward = self.evaluate(); print(f"[Evaluate Ep {ep}] Eval Raw Reward: {eval_raw_reward:.2f}"); log_dict["Evaluation/Raw Reward"] = eval_raw_reward
                    if eval_raw_reward >= 19:
                        self.best_reward = eval_raw_reward
                        model_path = os.path.join(self.save_dir, f"best_model_ep{ep}_rew{eval_raw_reward:.2f}.pt")
                        torch.save(self.q_net.state_dict(), model_path)
                        print(f"Saved new best model to {model_path}")
                wandb.log(log_dict, step=self.decision_steps)


    def evaluate(self, num_episodes=20): 
        total_rewards = []; collected_transitions = {'states': [], 'actions': [], 'rewards': [], 'next_states': [], 'dones': []}
        is_generating_data = getattr(self.args, 'generate_expert_data', False); target_transitions = getattr(self.args, 'num_expert_transitions', 0) if is_generating_data else 0; collected_count = 0
        test_preprocessor = AtariPreprocessor(frame_stack=self.args.frame_stack)
        if is_generating_data: print(f"Starting expert data generation. Target: {target_transitions} transitions."); self.q_net.eval()
        approx_steps_per_ep = self.max_episode_steps if self.max_episode_steps > 0 else 1000; eval_ep_count = num_episodes if not is_generating_data else int(target_transitions * 1.5 // approx_steps_per_ep) + 5
        for i in range(eval_ep_count): # ... (rest of evaluation loop as defined before) ...
            if is_generating_data and collected_count >= target_transitions: break
            obs, _ = self.test_env.reset(); state = test_preprocessor.reset(obs)
            if "Pong" in self.args.env_name: fire_obs, _, _, _, _ = self.test_env.step(1); state = test_preprocessor.step(fire_obs)
            done = False; episode_raw_reward = 0.0; decision_step_count = 0
            while not done:
                if is_generating_data and collected_count >= target_transitions: done = True; break
                if self.max_episode_steps > 0 and decision_step_count >= self.max_episode_steps: break
                current_decision_step_state_for_saving = np.array(state, dtype=np.uint8); action = self.select_action(state, use_greedy=True)
                accumulated_raw_reward = 0.0; frame_done = False; last_obs = None
                for _ in range(self.frame_skip): # ... (frame skip logic) ...
                    next_obs, reward, terminated, truncated, _ = self.test_env.step(action); accumulated_raw_reward += reward; frame_done = terminated or truncated; last_obs = next_obs
                    if frame_done: break
                if last_obs is not None: next_state = test_preprocessor.step(last_obs); next_state_for_saving = np.array(next_state, dtype=np.uint8)
                else: print("Warning: last_obs is None during eval/gen."); frame_done = True; next_state = state; next_state_for_saving = current_decision_step_state_for_saving
                if is_generating_data and last_obs is not None: # ... (save transition logic) ...
                    collected_transitions['states'].append(current_decision_step_state_for_saving); collected_transitions['actions'].append(action); collected_transitions['rewards'].append(accumulated_raw_reward)
                    collected_transitions['next_states'].append(next_state_for_saving); collected_transitions['dones'].append(1 if frame_done else 0); collected_count += 1
                    if collected_count % 1000 == 0: print(f"Collected {collected_count} / {target_transitions} expert transitions...")
                state = next_state; done = frame_done; episode_raw_reward += accumulated_raw_reward; decision_step_count += 1
            if not is_generating_data: total_rewards.append(episode_raw_reward)
        if is_generating_data: # ... (save expert data logic) ...
            print(f"\nFinished data generation phase. Collected {collected_count} transitions.")
            if collected_count >= target_transitions: 
                indices = list(range(collected_count)); 
                random.shuffle(indices); indices = indices[:target_transitions]; # ... (create np arrays) ...
                states_np=np.array([collected_transitions['states'][i] for i in indices],dtype=np.uint8); actions_np=np.array([collected_transitions['actions'][i] for i in indices],dtype=np.int64); rewards_np=np.array([collected_transitions['rewards'][i] for i in indices],dtype=np.float32); next_states_np=np.array([collected_transitions['next_states'][i] for i in indices],dtype=np.uint8); dones_np=np.array([collected_transitions['dones'][i] for i in indices],dtype=np.uint8)
                output_file = self.args.expert_data_output; np.savez_compressed(output_file, states=states_np, actions=actions_np, rewards=rewards_np, next_states=next_states_np, dones=dones_np); print(f"Saved {len(states_np)} transitions to {output_file}")
            else: print(f"Warning: Collected only {collected_count}, less than target {target_transitions}.")
            return {}
        return np.mean(total_rewards) if total_rewards else -21.0

    def train(self):
        if self.memory is None or len(self.memory) < self.batch_size or self.decision_steps < self.replay_start_size:
             return

        sample_result = self.memory.sample(self.batch_size)
        if sample_result is None: return
        states_np, actions, rewards, next_states_np, dones, is_experts, weights, indices, proportion_experts = sample_result
        # ----------------------------------------------------

        # Move essential tensors to device (unchanged)
        states = torch.from_numpy(states_np.astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(next_states_np.astype(np.float32)).to(self.device)
        actions, rewards, dones = actions.to(self.device), rewards.to(self.device), dones.to(self.device)
        is_experts, weights = is_experts.to(self.device), weights.to(self.device)

        # --- Calculate Losses (Element-wise - unchanged) ---
        loss_dq_elementwise = torch.zeros_like(rewards)
        loss_je_elementwise = torch.zeros_like(rewards)

        # --- >> 1. Agent Data Loss (J_DQ - C51 Loss) << --- (unchanged)
        agent_mask = ~is_experts
        if agent_mask.any():
            # ... (calculate target_distribution for agent) ...
            # ... (calculate elementwise C51 loss for agent) ...
             agent_indices = torch.where(agent_mask)[0]
             with torch.no_grad(): # ... (calculate target_distribution for agent data) ...
                target_next_logits=self.target_net(next_states[agent_mask]);target_next_probs=F.softmax(target_next_logits,dim=2);online_next_expected_q=self.q_net.get_expected_q_values(next_states[agent_mask]);next_best_actions=online_next_expected_q.argmax(1);target_next_best_action_probs=target_next_probs.gather(1,next_best_actions.view(-1,1,1).expand(-1,1,self.num_atoms)).squeeze(1);agent_rewards_n=rewards[agent_mask].view(-1,1);agent_dones_n=dones[agent_mask].view(-1,1);n_step_gamma=self.gamma**(self.n_step*self.frame_skip);Tz=agent_rewards_n+(1-agent_dones_n)*n_step_gamma*self.support.view(1,-1);Tz=Tz.clamp(self.vmin,self.vmax);b=(Tz-self.vmin)/self.delta_z;l=b.floor().long();u=b.ceil().long();l.clamp_(0,self.num_atoms-1);u.clamp_(0,self.num_atoms-1);eq_mask=(l==u);ne_mask=~eq_mask;m=torch.zeros_like(target_next_best_action_probs);batch_agent_size=agent_rewards_n.shape[0];offset=torch.linspace(0,((batch_agent_size-1)*self.num_atoms),batch_agent_size).unsqueeze(1).expand(batch_agent_size,self.num_atoms).long().to(self.device)
                if ne_mask.any(): p_l=(target_next_best_action_probs*(u.float()-b))[ne_mask];p_u=(target_next_best_action_probs*(b-l.float()))[ne_mask];l_offset_ne=(l+offset)[ne_mask];u_offset_ne=(u+offset)[ne_mask];m.view(-1).index_add_(0,l_offset_ne.view(-1),p_l.view(-1));m.view(-1).index_add_(0,u_offset_ne.view(-1),p_u.view(-1))
                if eq_mask.any(): l_offset_eq=(l+offset)[eq_mask];p_eq=target_next_best_action_probs[eq_mask];m.view(-1).index_add_(0,l_offset_eq.view(-1),p_eq.view(-1))
                target_distribution=m
             q_logits_agent = self.q_net(states[agent_mask]); action_logits_agent = q_logits_agent.gather(1, actions[agent_mask].view(-1, 1, 1).expand(-1, 1, self.num_atoms)).squeeze(1); log_pred_probs = F.log_softmax(action_logits_agent, dim=1); loss_dq_elementwise.scatter_(0, agent_indices, -(target_distribution * log_pred_probs).sum(1))


        # --- >> 2. Expert Data Loss (J_E - Supervised Margin Loss) << --- (unchanged)
        expert_mask = is_experts
        if expert_mask.any():
             # ... (calculate elementwise supervised loss for expert) ...
             expert_indices = torch.where(expert_mask)[0]; q_logits_expert = self.q_net(states[expert_mask]); q_exp_expert = (F.softmax(q_logits_expert, dim=2) * self.support).sum(2); expert_actions_batch = actions[expert_mask]; q_exp_expert_action = q_exp_expert.gather(1, expert_actions_batch.unsqueeze(1)).squeeze(1); action_mask = torch.ones_like(q_exp_expert).bool(); action_mask.scatter_(1, expert_actions_batch.unsqueeze(1), False); q_exp_masked = torch.where(action_mask, q_exp_expert, torch.tensor(-float('inf'), device=self.device)); max_non_expert_q = q_exp_masked.max(1).values; loss_je = F.relu(max_non_expert_q + self.args.margin - q_exp_expert_action); loss_je_elementwise.scatter_(0, expert_indices, loss_je)

        # --- >> 3. L2 Regularization Loss (J_L2) << --- (unchanged)
        loss_l2 = torch.tensor(0.0, device=self.device)
        if self.args.lambda2 > 0:
            for param in self.q_net.parameters():
                if param.ndim > 1: loss_l2 += torch.sum(param ** 2)

        # --- >> 4. Combine Losses << --- (unchanged)
        total_loss = (weights * (loss_dq_elementwise + self.args.lambda1 * loss_je_elementwise)).mean() + self.args.lambda2 * loss_l2

        # --- Optimization --- (unchanged)
        self.optimizer.zero_grad(); total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.args.grad_clip); self.optimizer.step()

        # --- Update PER Priorities --- (unchanged - uses the modified buffer method)
        priorities_update = loss_dq_elementwise + self.args.lambda1 * loss_je_elementwise
        priorities_update_np = priorities_update.abs().detach().cpu().numpy()
        is_expert_flags_np = is_experts.cpu().numpy() # Get boolean flags as numpy array
        self.memory.update_priorities(indices, priorities_update_np, is_expert_flags_np)

        self.train_count += 1

        # --- Target Network Update --- (unchanged)
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # --- Logging ---
        if self.train_count % 100 == 0: # Log every 100 training steps
            if not getattr(self.args, 'generate_expert_data', False):
                log_data = {"Training/Total Loss": total_loss.item(), "Training/L2 Loss": loss_l2.item()}
                if agent_mask.any(): log_data["Training/DQ Loss Mean"] = loss_dq_elementwise[agent_mask].mean().item()
                if expert_mask.any(): log_data["Training/JE Loss Mean"] = loss_je_elementwise[expert_mask].mean().item()
                # <<< Log expert proportion >>>
                log_data["Sampling/Proportion Experts"] = proportion_experts
                # <<< Log PER Beta value >>>
                log_data["Parameters/PER Beta"] = self.memory.beta

                wandb.log(log_data, step=self.decision_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5")
    parser.add_argument("--frame-stack", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--max-episode-steps", type=int, default=27000 // 4)
    parser.add_argument("--episodes", type=int, default=350)
    parser.add_argument("--replay-start-size", type=int, default=20000) # Increased
    parser.add_argument("--train-per-decision-step", type=int, default=1)
    parser.add_argument("--lr", type=float, default=6.25e-5)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--target-update-frequency", type=int, default=2000)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--vmin", type=float, default=-5.0) # Adjusted
    parser.add_argument("--vmax", type=float, default=5.0)  # Adjusted
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay-steps", type=int, default=500000) # Increased
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--per-alpha", type=float, default=0.5)
    parser.add_argument("--per-beta0", type=float, default=0.4)
    parser.add_argument("--per-beta-steps", type=int, default=1000000) # Increased
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--save-dir", type=str, default="./dqfd_results") # Changed dir
    parser.add_argument("--wandb-project", type=str, default="DLP-Lab5-DuelingC51-Pong", help="WandB project name")
    parser.add_argument("--wandb-run-name", type=str, default="rainbow_DQFD", help="Base WandB run name")
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--save-frequency", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=777)

    # --- <<< DQFD Specific Args >>> ---
    parser.add_argument("--load-expert-data", type=str, default=None, help="Path to expert data (.npz file). If None, runs as standard agent.")
    parser.add_argument("--pretrain-steps", type=int, default=1000000, help="Number of pre-training steps using expert data.")
    parser.add_argument("--lambda1", type=float, default=1.0, help="Weight for expert supervised loss (JE).")
    parser.add_argument("--lambda2", type=float, default=1e-5, help="Weight for L2 regularization loss (JL2).")
    parser.add_argument("--margin", type=float, default=1, help="Margin 'l' for supervised loss JE.")
    parser.add_argument("--per-epsilon", type=float, default=1e-6, help="Epsilon added to agent priorities in PER.")
    parser.add_argument("--expert-epsilon", type=float, default=1.0, help="Epsilon added to expert priorities in PER (should be > per-epsilon).") # Changed default

    # --- Generation Args (Unchanged) ---
    parser.add_argument("--generate-expert-data", action='store_true')
    parser.add_argument("--load-expert-model", type=str, default=None)
    parser.add_argument("--expert-data-output", type=str, default="expert_data_c51.npz")
    parser.add_argument("--num-expert-transitions", type=int, default=50000)


    # <<< New Args for Early Stopping Pre-training >>>
    parser.add_argument("--val-split-ratio", type=float, default=0.1, help="Fraction of expert data to use for validation (e.g., 0.1 for 10%). Set to 0 to disable validation.")
    parser.add_argument("--validate-every", type=int, default=1000, help="Frequency (in pre-training steps) to check validation loss.")
    parser.add_argument("--patience", type=int, default=20, help="Number of validation checks without improvement before early stopping pre-training. Set to 0 to disable early stopping.")
    parser.add_argument("--min-delta", type=float, default=1e-5, help="Minimum improvement in validation loss to reset patience.")
    parser.add_argument("--pretrain-lr", type=float, default=8e-6, help="Learning rate specifically for pre-training phase. If None, uses main --lr.")
    parser.add_argument("--accuracy-threshold", type=float, default=0.80, help="Minimum validation accuracy required before checking loss patience for early stopping.")


    args = parser.parse_args()

    # Argument Validation & WandB Init (adjusted)
    if args.vmin >= args.vmax: raise ValueError("--vmin must be strictly less than --vmax")
    if not args.generate_expert_data:
        if args.load_expert_data and not (0 <= args.val_split_ratio < 1): raise ValueError("--val-split-ratio must be between 0 and 1 (exclusive of 1).")
        use_es = args.load_expert_data and args.patience > 0 and args.val_split_ratio > 0
        if use_es and args.accuracy_threshold <= 0: print("Warning: Early stopping enabled but accuracy_threshold <= 0. Loss patience check will start immediately.")
        if args.load_expert_data and args.expert_epsilon <= args.per_epsilon: print(f"Warning: --expert-epsilon ({args.expert_epsilon}) should ideally be greater than --per-epsilon ({args.per_epsilon}) for DQFD.")
        args.replay_start_size=max(args.batch_size*args.n_step,args.replay_start_size);args.target_update_frequency=max(1,args.target_update_frequency);args.per_beta_steps=max(0,args.per_beta_steps);dqfd_tag="DQFD" if args.load_expert_data else "NoDQFD";es_tag=f"_ES{args.patience}Acc{args.accuracy_threshold:.0%}" if use_es else "";ptlr_tag=f"_ptLR{args.pretrain_lr}" if args.pretrain_lr is not None else ""; args.wandb_run_name=f"{dqfd_tag}_fs{args.frame_skip}_n{args.n_step}_lr{args.lr}{ptlr_tag}_b{args.batch_size}_atoms{args.num_atoms}_v{args.vmin}-{args.vmax}_pt{args.pretrain_steps}{es_tag}_l1_{args.lambda1}_seed{args.seed}";
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args), save_code=True)
        # Define metrics after init
        if args.load_expert_data and args.pretrain_steps > 0:
            wandb.define_metric("Pretrain/Step")
            wandb.define_metric("Pretrain/*", step_metric="Pretrain/Step")


    # Seeds, Agent Instantiation, Execute Mode (Unchanged)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    agent = DQNAgent(env_name=args.env_name, args=args)
    if args.generate_expert_data: # ... (generation logic) ...
        print("--- Running in Expert Data Generation Mode ---");
        if args.load_expert_model is None or not os.path.exists(args.load_expert_model): raise ValueError("--load-expert-model required."); print(f"Loading model: {args.load_expert_model}"); agent.q_net.load_state_dict(torch.load(args.load_expert_model, map_location=agent.device)); print("Model loaded."); agent.evaluate(num_episodes=-1); print("--- Generation Finished ---")
    else:
        print("--- Running in Training Mode ---"); agent.run(args.episodes);
        # Ensure wandb finishes cleanly
        if wandb.run is not None:
             wandb.finish()
        print("--- Training Finished ---")