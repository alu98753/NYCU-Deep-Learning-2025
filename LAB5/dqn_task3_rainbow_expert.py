# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import ale_py
from collections import deque
import wandb
import argparse
import time
from torch.nn.parameter import Parameter # Added import
import os, random, argparse, cv2
from typing import Deque, Dict, List, Tuple
import matplotlib.pyplot as plt
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

class SegmentTree:
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(self, capacity: int, operation: Callable, init_value: float):
        """Initialization.

        Args:
            capacity (int)
            operation (function)
            init_value (float)

        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self, start: int = 0, end: int = 0) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self, idx: int, val: float):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]

class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self, start: int = 0, end: int = 0) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self, upperbound: float) -> int:
        """Find the highest index `i` about upper bound in the tree"""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity

class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialization.

        Args:
            capacity (int)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self, start: int = 0, end: int = 0) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)

# Function to initialize Conv weights (NoisyLinear handles its own init)
def init_weights(m):
    if isinstance(m, nn.Conv2d): # Only apply to Conv2d now
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self, 
        obs_shape: tuple, 
        size: int, 
        batch_size: int = 32, 
        n_step: int = 1, 
        gamma: float = 0.99
    ):
        correct_shape = (size, *obs_shape) # 例如變成 (100000, 4, 84, 84)

        self.obs_buf = np.zeros(correct_shape, dtype=np.uint8)
        self.next_obs_buf = np.zeros(correct_shape, dtype=np.uint8)
        self.acts_buf = np.zeros([size], dtype=np.int64)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: np.ndarray,
        act: int, # <--- 改為 int
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return dict(
            obs=torch.as_tensor(self.obs_buf[idxs], device=device, dtype=torch.float32),
            next_obs=torch.as_tensor(self.next_obs_buf[idxs], device=device, dtype=torch.float32),
            acts=torch.as_tensor(self.acts_buf[idxs], device=device, dtype=torch.int64).unsqueeze(-1),
            rews=torch.as_tensor(self.rews_buf[idxs], device=device, dtype=torch.float32).unsqueeze(-1),
            done=torch.as_tensor(self.done_buf[idxs], device=device, dtype=torch.float32).unsqueeze(-1),
            indices=torch.as_tensor(idxs, device=device)
        )

    
    def sample_batch_from_idxs(self, idxs: np.ndarray) -> Dict[str, torch.Tensor]: # Return torch.Tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return dict(
            obs=torch.as_tensor(self.obs_buf[idxs], device=device, dtype=torch.uint8), # Return uint8 tensor
            next_obs=torch.as_tensor(self.next_obs_buf[idxs], device=device, dtype=torch.uint8),
            acts=torch.as_tensor(self.acts_buf[idxs], device=device, dtype=torch.int64).unsqueeze(-1), # Already int64
            rews=torch.as_tensor(self.rews_buf[idxs], device=device, dtype=torch.float32).unsqueeze(-1),
            done=torch.as_tensor(self.done_buf[idxs], device=device, dtype=torch.float32).unsqueeze(-1),
            # No 'indices' or 'weights' needed for N-step loss calculation typically
        )
    
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):

    """Prioritized Replay buffer.
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
    """    
    def __init__(
        self, 
        obs_shape: tuple, 
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6,
        n_step: int = 1, 
        gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(
            obs_shape, size, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray,
        act: int, # <--- 改為 int
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)
        
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, torch.Tensor]: # <--- 修改返回類型提示為 Tensor
        """Sample a batch of experiences and return them as tensors."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional() # Get indices (List[int])
        indices_np = np.array(indices) # Convert to numpy array for indexing buffer

        # Calculate weights as numpy array first
        weights_np = np.array([self._calculate_weight(i, beta) for i in indices])

        # --- **** 將 NumPy 陣列轉換為 PyTorch Tensors **** ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        obs_tensor = torch.as_tensor(self.obs_buf[indices_np], device=device, dtype=torch.uint8)
        next_obs_tensor = torch.as_tensor(self.next_obs_buf[indices_np], device=device, dtype=torch.uint8)
        # 使用 int64 作為動作索引的類型
        acts_tensor = torch.as_tensor(self.acts_buf[indices_np], device=device, dtype=torch.int64).unsqueeze(-1)
        rews_tensor = torch.as_tensor(self.rews_buf[indices_np], device=device, dtype=torch.float32).unsqueeze(-1)
        done_tensor = torch.as_tensor(self.done_buf[indices_np], device=device, dtype=torch.float32).unsqueeze(-1)
        weights_tensor = torch.as_tensor(weights_np, dtype=torch.float32, device=device).unsqueeze(-1) # 轉換 weights

        return dict(
            obs=obs_tensor,
            next_obs=next_obs_tensor,
            acts=acts_tensor,
            rews=rews_tensor,
            done=done_tensor,
            weights=weights_tensor, # 返回 Tensor
            indices=indices_np,     # indices 保持為 numpy array，因為 update_priorities 需要它
        )
       
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    (其他屬性說明不變)
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.1): # 你的 std_init 是 0.1
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        # --- 修正 bias_epsilon 的生成 ---
        # 原始 NoisyNet 論文的 bias epsilon 應該是獨立採樣的
        self.bias_epsilon.copy_(self.scale_noise(self.out_features)) # 使用獨立的 scale_noise
        # --- 修正結束 ---

    # --- **** 修改 forward **** ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        if self.training: # 檢查是否處於訓練模式
            # 使用學習到的 sigma 和採樣的 epsilon 計算帶噪聲的權重和偏差
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else: # 處於評估模式 (eval)
            # 只使用均值權重和偏差，不加噪聲
            weight = self.weight_mu
            bias = self.bias_mu
        # --- **** 修改結束 **** ---

        return F.linear(x, weight, bias)

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        # --- 將生成移到正確的設備 ---
        # 需要知道 weight_mu 所在的設備，但靜態方法無法直接訪問 self
        # 一種方法是傳入 device，或者在調用處處理
        # 暫時保持原樣，但在 reset_noise 中確保 scale_noise 返回的張量在正確設備上
        # （實際上 torch.randn 預設在 cpu，需要 .to(device)）
        # 更簡潔的方式是在 reset_noise 中處理
        # x = torch.randn(size) # 原寫法
        # return x.sign().mul(x.abs().sqrt())
        # 在 reset_noise 中處理設備即可，這裡保持不變
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    # --- 在 reset_noise 中修正 scale_noise 的設備問題 ---
    def reset_noise(self):
        """Make new noise on the correct device."""
        device = self.weight_mu.device # 獲取參數所在的設備
        epsilon_in = self.scale_noise(self.in_features).to(device)
        epsilon_out = self.scale_noise(self.out_features).to(device)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        # 使用獨立的 scale_noise，並確保在正確設備上
        self.bias_epsilon.copy_(self.scale_noise(self.out_features).to(device))

# no dueling
# class Network(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int):
#         """Initialization."""
#         super(Network, self).__init__()

#         self.feature = nn.Linear(in_dim, 128)
#         self.noisy_layer1 = NoisyLinear(128, 128)
#         self.noisy_layer2 = NoisyLinear(128, out_dim)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward method implementation."""
#         feature = F.relu(self.feature(x))
#         hidden = F.relu(self.noisy_layer1(feature))
#         out = self.noisy_layer2(hidden)
        
#         return out
    
#     def reset_noise(self):
#         """Reset all noisy layers."""
#         self.noisy_layer1.reset_noise()
#         self.noisy_layer2.reset_noise()

class DQN(nn.Module):
    """
    Dueling Double Deep Q Network with Noisy Linear Layers % PER.
    """
    def __init__(self, num_actions: int, frame_stack: int = 4,
                 use_dueling: bool = True, use_noisy: bool = True, # Assume Dueling and Noisy are True
                 noisy_std_init: float = 0.5):
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.frame_stack = frame_stack
        self.use_dueling = use_dueling
        self.use_noisy = use_noisy

        linear_layer = lambda in_f, out_f: NoisyLinear(in_f, out_f, std_init=noisy_std_init) if use_noisy else nn.Linear(in_f, out_f)

        self.feature_layer = nn.Sequential(
            nn.Conv2d(self.frame_stack, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
             dummy_input = torch.zeros(1, frame_stack, 84, 84)
             feature_output = self.feature_layer(dummy_input)
             self.feature_dim = feature_output.shape[1]

        if self.use_dueling:
            self.advantage_hidden_layer = linear_layer(self.feature_dim, 512)
            # --- Output size is num_actions ---
            self.advantage_layer = linear_layer(512, num_actions)

            self.value_hidden_layer = linear_layer(self.feature_dim, 512)
            # --- Output size is 1 ---
            self.value_layer = linear_layer(512, 1)
        else:
            self.common_hidden_layer = linear_layer(self.feature_dim, 512)
            # --- Output size is num_actions ---
            self.final_layer = linear_layer(512, num_actions)

        # Apply Kaiming init only if not using NoisyLinear for FC layers
        if not use_noisy:
             self.apply(init_weights)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Calculates Q-values. """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.0 and x.max() <= 255.0: # Check if likely uint8 scale
            x = x / 255.0

        feature = self.feature_layer(x)

        if self.use_dueling:
            adv_hid = F.relu(self.advantage_hidden_layer(feature))
            val_hid = F.relu(self.value_hidden_layer(feature))

            advantage = self.advantage_layer(adv_hid) # Shape: (batch, num_actions)
            value = self.value_layer(val_hid)         # Shape: (batch, 1)

            # Combine value and advantage streams (Equation 9 in Dueling DQN paper)
            # Q(s,a) = V(s) + (A(s,a) - mean_a'(A(s,a')))
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True) # Shape: (batch, num_actions)
        else:
            common_hid = F.relu(self.common_hidden_layer(feature))
            q_values = self.final_layer(common_hid) # Shape: (batch, num_actions)

        return q_values # Return Q-values directly

    def reset_noise(self):
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

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


class DQNAgent:
    """DQN Agent interacting with environment.
    
    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        q_net (Network): model to train and select actions
        target_net (Network): target model to update
        optimizer (torch.optim): optimizer for training q_net
        transition (list): transition information including
                           state, action, reward, next_state, done
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(self, env_name="ALE/Pong-v5", args=None):
        """Initialization.
        
        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is use
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
                        n_step (int): step number to calculate n-step td error
        """
        # NoisyNet: All attributes related to epsilon are removed
        self.args = args
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.preprocessor = AtariPreprocessor()
        self.frame_skip = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Using device:", self.device)
        # 做一次 dummy reset 來獲取預處理後的 shape
        dummy_obs_raw, _ = self.env.reset()
        dummy_state_processed = self.preprocessor.reset(dummy_obs_raw)
        obs_shape = dummy_state_processed.shape # 例如 (4, 84, 84)
        self.action_dim = self.env.action_space.n
        
        self.batch_size = args.batch_size
        self.target_update = args.target_update
        self.seed = args.seed
        self.initial_beta = args.beta
        self.alpha = args.alpha
        self.beta = args.beta
        self.prior_eps = args.prior_eps
        
        self.gamma = args.gamma
        self.n_step = args.n_step
        self.memory_size = args.memory_size
        self.noisy_std_init = args.noisy_std_init
        self.beta_annealing_steps = args.beta_annealing_steps # Get from args

        # PER
        # memory for 1-step Learning
        self.prior_eps = args.prior_eps
        self.memory = PrioritizedReplayBuffer(obs_shape, self.memory_size, self.batch_size, alpha=self.alpha, gamma=self.gamma)
        # memory for N-step Learning
        self.use_n_step = True if self.n_step > 1 else False
        if self.use_n_step:
            self.n_step = args.n_step
            self.memory_n = ReplayBuffer(
                obs_shape, self.memory_size, self.batch_size, n_step=self.n_step, gamma=self.gamma
            )
            
        # --- Expert Data Pre-filling ---
        if args.expert_data_path and args.num_expert_transitions > 0:
            self.load_and_prefill_expert_data(args.expert_data_path, args.num_expert_transitions)
        else:
            print("No expert data path provided or num_expert_transitions is 0. Skipping pre-filling.")
        # ---
        
        # networks: q_net, target_net
        self.q_net = DQN(
            num_actions=self.action_dim, frame_stack=args.frame_stack,
            use_dueling=True, use_noisy=True, # Set based on desired config
            noisy_std_init=self.noisy_std_init
        ).to(self.device)
        self.target_net = DQN(
            num_actions=self.action_dim, frame_stack=args.frame_stack,
            use_dueling=True, use_noisy=True,
            noisy_std_init=self.noisy_std_init
        ).to(self.device)
        self.q_net.feature_layer.apply(init_weights)
        # Optional Compilation
        if args.compile:
             print("Compiling networks...")
             try:
                 self.q_net = torch.compile(self.q_net, mode="reduce-overhead")
                 self.target_net = torch.compile(self.target_net, mode="reduce-overhead")
                 print("Compilation successful.")
             except Exception as e:
                 print(f"Warning: Network compilation failed: {e}. Proceeding without compilation.")
                 self.args.compile = False # Disable compile flag if it fails

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        # optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.learning_rate if hasattr(args, 'learning_rate') else 1e-4) # 使用 lr from args
        # Optional LR Scheduler
        self.lr_scheduler = None
        if args.lr_decay_steps > 0:
             print(f"Using Linear LR decay over {args.lr_decay_steps} steps from {args.learning_rate} to {args.lr_end}")
             self.lr_scheduler = optim.lr_scheduler.LinearLR(
                 self.optimizer, start_factor=1.0, end_factor=args.lr_end/args.learning_rate, total_iters=args.lr_decay_steps
             )
        
        # mode: train / test
        self.is_test = False
        
        self.env_count = 0
        self.best_reward = -21 # Initialized for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        # self.train_per_step = args.train_per_step
        self.early_stop_counter = 0 
        self.early_stop_threshold = 5 # Early stop threshold

        self.save_dir = os.path.join(
            args.save_dir,
            f"{time.strftime('%Y%m%d-%H%M%S')}_{args.wandb_run_name}"
            )
        os.makedirs(self.save_dir, exist_ok=True)

    def load_expert_data(self, path, num_to_load):
        """Loads expert transitions from a pickle file."""
        print(f"Loading expert data from {path}...")
        try:
            with open(path, 'rb') as f:
                all_transitions = pickle.load(f)
            print(f"Loaded {len(all_transitions)} transitions.")
            # Ensure data is in the correct format (s, a, r, s', d)
            # Add checks here if necessary based on how data was saved
            return all_transitions[:num_to_load]
        except FileNotFoundError:
            print(f"Expert data file not found: {path}")
            return []
        except Exception as e:
            print(f"Error loading expert data: {e}")
            return []

    def load_and_prefill_expert_data(self, path, num_to_fill):
        """Loads expert data and pre-fills the PER buffer directly."""
        expert_transitions = self.load_expert_data(path, num_to_fill)

        print(f"Pre-filling replay buffer with up to {num_to_fill} expert 1-step transitions...")
        added_count = 0
        if expert_transitions:
            initial_max_priority = self.memory.max_priority # Should be 1.0 initially

            for obs, act, rew, next_obs, done in expert_transitions:
                if added_count >= num_to_fill:
                    break

                # Directly write to underlying numpy buffers
                ptr = self.memory.ptr
                self.memory.obs_buf[ptr] = obs.astype(np.uint8) # Ensure uint8
                self.memory.next_obs_buf[ptr] = next_obs.astype(np.uint8) # Ensure uint8
                self.memory.acts_buf[ptr] = int(act) # Ensure int
                self.memory.rews_buf[ptr] = float(rew)
                self.memory.done_buf[ptr] = float(done) # Ensure float (0.0 or 1.0)

                # Directly set priority in segment trees
                priority_value = (initial_max_priority + self.prior_eps) ** self.alpha # Use initial max_priority + eps
                self.memory.sum_tree[ptr] = priority_value
                self.memory.min_tree[ptr] = priority_value

                # Update pointers and size
                self.memory.ptr = (ptr + 1) % self.memory.max_size
                self.memory.size = min(self.memory.size + 1, self.memory.max_size)
                self.memory.tree_ptr = self.memory.ptr # Keep tree_ptr in sync

                added_count += 1

            # After filling, ensure max_priority reflects the added values if needed
            # self.memory.max_priority = max(self.memory.max_priority, initial_max_priority)

        print(f"Finished pre-filling with {added_count} transitions. Buffer size: {len(self.memory)}")


    def select_action(self, state_tensor: torch.Tensor) -> np.ndarray: # 接收 Tensor
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        state_tensor = state_tensor.to(self.device) 
        with torch.no_grad(): # 在推斷時不需要計算梯度
            q_values = self.q_net(state_tensor) # q_net.forward 會處理歸一化
   
        selected_action = q_values.argmax(dim=1) # 取最大 Q 值的動作索引
        selected_action = selected_action.item() # 轉換為 Python int
        
        return selected_action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        
        if not self.is_test:
            self.transition += [reward, next_state, done]
            
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    
        return next_state, reward, done

    def env_step(self, action):
        """step function for training."""
        total_reward = 0.0
        last_raw_obs = None
        done = False # 初始化 done
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.frame_skip):
            next_obs_raw, reward, terminated, truncated, info = self.env.step(action)
            last_raw_obs = next_obs_raw # 記錄最後的原始觀測

            total_reward += reward
            done = terminated or truncated
            if done:
                break
        next_state_processed = self.preprocessor.step(last_raw_obs)

        return next_state_processed, total_reward, done

    def test_env_step(self, action):
        """step function for testing, returns processed state."""
        total_reward = 0.0
        last_raw_obs = None
        done = False # 初始化 done
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.frame_skip):
            next_obs_raw, reward, terminated, truncated, info = self.test_env.step(action)
            last_raw_obs = next_obs_raw
            total_reward += reward
            done = terminated or truncated
            if done:
                break
        # 使用 preprocessor 處理最後的原始觀測，得到堆疊狀態
        next_state_processed = self.preprocessor.step(last_raw_obs)
        return next_state_processed, total_reward, done # 返回處理後的狀態和 done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta) # Samples are now tensors

        weights = samples["weights"].to(self.device) # Already tensor on device
        indices = samples["indices"] # Numpy indices

        # 1-step Learning loss
        # Pass samples dict directly if _compute_dqn_loss handles tensors
        elementwise_loss_1 = self._compute_dqn_loss(samples, self.gamma)
        loss = torch.mean(elementwise_loss_1 * weights) # Weighted loss

        total_elementwise_loss = elementwise_loss_1 # Keep track for priority update

        # N-step Learning loss
        if self.use_n_step:
            gamma_n = self.gamma ** self.n_step
            # Fetch n-step samples using indices, should return tensors
            samples_n = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n = self._compute_dqn_loss(samples_n, gamma_n)
            # Combine losses (simple addition, could use weighting)
            # The elementwise loss for priority should probably reflect both
            total_elementwise_loss = elementwise_loss_1 + elementwise_loss_n # Combine elementwise losses for priority
            # The final loss combines weighted 1-step and N-step elementwise losses
            loss = torch.mean(total_elementwise_loss * weights) # Recompute weighted average

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.q_net.parameters(), self.args.gradient_clip) # Use clip value from args
        self.optimizer.step()

        # Optional LR scheduler step
        if self.lr_scheduler and self.env_count < self.args.lr_decay_steps:
             self.lr_scheduler.step()


        # PER: update priorities using combined elementwise loss
        loss_for_prior = total_elementwise_loss.detach().cpu().numpy().squeeze() # Use combined loss, remove extra dim
        new_priorities = np.abs(loss_for_prior) + self.prior_eps # Use absolute error
        self.memory.update_priorities(indices, new_priorities) # Pass numpy arrays

        # NoisyNet: reset noise only for the online network
        self.q_net.reset_noise()
        # Removed target_net.reset_noise()

        return loss.item()

    # --- _compute_dqn_loss (修改以處理 Tensor 輸入) ---
    def _compute_dqn_loss(self, samples: Dict[str, torch.Tensor], gamma: float) -> torch.Tensor:
        """Return elementwise dqn loss."""
        device = self.device
        state = samples["obs"].to(device).float() / 255.0
        next_state = samples["next_obs"].to(device).float() / 255.0
        action = samples["acts"].to(device) # Already LongTensor on device
        reward = samples["rews"].to(device) # Already FloatTensor on device
        done = samples["done"].to(device) # Already FloatTensor on device

        curr_q_value = self.q_net(state).gather(1, action)

        with torch.no_grad():
            next_action = self.q_net(next_state).argmax(1, keepdim=True)
            next_q = self.target_net(next_state).gather(1, next_action)
            target = reward + gamma * next_q * (1 - done)

        # Use Huber loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction='none', beta=1.0)
        return elementwise_loss
        
    def train(self, num_episodes: int, eval_episode_interval: int): # Removed target_total_steps
        """Train the agent for a specified number of episodes."""
        self.is_test = False

        update_cnt = 0
        losses = []
        total_steps = self.env_count # Continue counting from pre-fill if any

        print(f"🚀 Starting training for {num_episodes} episodes...")
        start_time = time.time()

        for episode_idx in range(1, num_episodes + 1):
            obs_raw, _ = self.env.reset(seed=self.seed + episode_idx)
            state = self.preprocessor.reset(obs_raw)
            score = 0
            done = False
            episode_steps = 0

            self.q_net.train() # Ensure train mode for the episode

            while not done:
                # --- Action Selection ---
                # No need for inference_mode here if using Noisy Nets for exploration
                state_tensor = torch.from_numpy(state).to(self.device, dtype=torch.float32, non_blocking=True).unsqueeze(0)
                action = self.select_action(state_tensor) # Uses q_net in train mode internally
                # ---

                # --- Environment Step ---
                next_state, reward, done = self.env_step(action)
                score += reward
                # ---

                # --- Store Transition ---
                # Store 1-step transition in PER buffer
                self.memory.store(state, action, reward, next_state, done)
                # Store 1-step transition in N-step buffer's internal deque
                if self.use_n_step:
                    self.memory_n.store(state, action, reward, next_state, done) # This will manage its internal n-step buffer
                # ---

                state = next_state
                total_steps += 1
                self.env_count = total_steps
                episode_steps += 1

                # --- PER Beta Annealing ---
                fraction = min(total_steps / self.beta_annealing_steps, 1.0)
                self.beta = self.initial_beta + fraction * (1.0 - self.initial_beta)
                # ---

                # --- Model Update ---
                if len(self.memory) >= self.replay_start_size and total_steps % self.args.train_frequency == 0:
                    # Perform `gradient_steps` updates
                    for _ in range(self.args.gradient_steps):
                        if len(self.memory) >= self.batch_size: # Check if enough samples for a batch
                            loss = self.update_model()
                            if loss is not None: # update_model might return None if sampling fails
                                losses.append(loss)
                                update_cnt += 1
                                # --- Target Network Update ---
                                if update_cnt % self.target_update == 0:
                                    self._target_hard_update()
                                # ---
                                # --- Log Loss ---
                                if update_cnt % 1000 == 0 and wandb.run: # Log every 1000 updates
                                    current_lr = self.optimizer.param_groups[0]['lr']
                                    wandb.log({
                                        'Loss': loss, 'Beta': self.beta, 'Learning Rate': current_lr,
                                        'Q-Net/Weight Sigma Mean': self.get_noisy_sigma_mean(self.q_net)
                                    }, step=total_steps)
                        # ---

                # --- Episode Truncation ---
                if episode_steps >= self.max_episode_steps:
                    print(f"Episode {episode_idx} truncated at max steps {self.max_episode_steps}.")
                    done = True
                # ---

            # --- End of Episode ---
            avg_loss = np.mean(losses[-episode_steps:]) if losses else 0 # Avg loss for this episode
            print(f"[Train] Ep:{episode_idx}/{num_episodes} | Score:{score:.2f} | Steps:{episode_steps} | Total Steps:{total_steps} | Avg Loss:{avg_loss:.4f} | Buffer:{len(self.memory)}")
            if wandb.run:
                wandb.log({
                    'Episode': episode_idx,
                    'Total Reward': score,
                    'Episode Steps': episode_steps,
                }, step=total_steps)
            # ---

            # --- Periodic Evaluation ---
            if episode_idx > 0 and episode_idx % eval_episode_interval == 0:
                eval_reward = self.evaluate() # Evaluate handles logging internally
                # --- Early Stopping Logic ---
                if eval_reward >= 19:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.early_stop_threshold:
                        print(f"[Early Stop] Eval Reward >= 19 for {self.early_stop_threshold} consecutive evaluations. Stopping training at episode {episode_idx}.")
                        break
                else:
                    self.early_stop_counter = 0
                # ---
                self.q_net.train() # Ensure back to train mode after evaluation
            # ---

            # --- Periodic Checkpoint Saving ---
            # Save based on total steps, maybe adjust frequency
            save_interval = 100000 # Example: save every 100k steps
            if 'last_save_step' not in locals(): last_save_step = -save_interval # Initialize
            if total_steps >= last_save_step + save_interval:
                model_path = os.path.join(self.save_dir, f"checkpoint_step_{total_steps}.pt")
                self.save_checkpoint(model_path) # Use a helper function
                last_save_step = total_steps
            # ---

        self.env.close()
        final_model_path = os.path.join(self.save_dir, "final_model.pt")
        self.save_checkpoint(final_model_path)
        print(f"Training finished. Final model saved to {final_model_path}")

    # --- evaluate (使用修正後的版本) ---
    def evaluate(self, n_episodes=30):
        """Evaluate the agent's performance."""
        print("Starting evaluation...")
        self.q_net.eval() # Set network to evaluation mode

        total_rewards = []
        for ep in range(n_episodes):
            obs_raw, _ = self.test_env.reset(seed=self.seed + ep + 1000) # Use different seeds for eval
            state = self.preprocessor.reset(obs_raw)
            done = False
            episode_reward = 0
            ep_steps = 0

            while not done:
                with torch.inference_mode():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device, non_blocking=True)
                    # Select action using deterministic policy (q_net is in eval mode)
                    action = self.select_action(state_tensor)

                next_state, reward, done = self.test_env_step(action)
                episode_reward += reward
                state = next_state
                ep_steps += 1
                if ep_steps >= self.max_episode_steps:
                    # print(f"Evaluation episode {ep+1} truncated at {ep_steps} steps.")
                    break

            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        print(f"Evaluation finished. Average Reward over {n_episodes} episodes: {avg_reward:.2f}")

        # Save best model checkpoint
        if avg_reward >= self.best_reward and avg_reward >= 19: # Pong target
             self.best_reward = avg_reward
             best_model_path = os.path.join(self.save_dir, f"best_model_step_{self.env_count}_reward_{avg_reward:.2f}.pt")
             self.save_checkpoint(best_model_path, is_best=True)

        print(f"[Eval] EnvS: {self.env_count} Eval Reward: {avg_reward:.2f} Best: {self.best_reward:.2f}")

        if wandb.run:
            wandb.log({
                "Eval Reward": avg_reward,
                "Best Eval Reward": self.best_reward
            }, step=self.env_count)

        self.q_net.train() # Set network back to training mode
        return avg_reward

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.target_net.load_state_dict(self.q_net.state_dict())
                
    # --- Helper for saving checkpoints ---
    def save_checkpoint(self, path, is_best=False):
         """Saves model checkpoint."""
         print(f"Saving {'best ' if is_best else ''}checkpoint to {path}")
         # If using torch.compile, save the original model's state_dict
         model_state = self.q_net._orig_mod.state_dict() if hasattr(self.q_net, '_orig_mod') else self.q_net.state_dict()
         torch.save(model_state, path)

    # --- Helper for logging noisy sigma (optional) ---
    def get_noisy_sigma_mean(self, network):
        """Calculates the mean absolute value of sigma parameters in NoisyLinear layers."""
        sigmas = []
        for module in network.modules():
            if isinstance(module, NoisyLinear):
                sigmas.append(module.weight_sigma.abs().mean().item())
                sigmas.append(module.bias_sigma.abs().mean().item())
        return np.mean(sigmas) if sigmas else 0

# --- 主程式 (`if __name__ == "__main__":`) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rainbow DQN (Non-Distributional) Training")
    seed = 777

    # --- Seeding ---
    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) # for multi-GPU
            # Set deterministic options for reproducibility if needed (might slow down)
            # torch.backends.cudnn.benchmark = False
            # torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed) # Also seed Python's random module
    seed_torch(seed)
    # --- Performance options ---
    torch.backends.cudnn.benchmark = True # Enable benchmark mode for potential speedup
    torch.backends.cuda.matmul.allow_tf32 = True # Enable TF32 for matmul
    torch.backends.cudnn.allow_tf32 = True # Enable TF32 for cuDNN

    # --- Argument Parser ---
    # Basic Training Params
    # --- Argument Parser (使用底線定義參數) ---
    parser = argparse.ArgumentParser(description="Rainbow DQN (Non-Distributional) Training")
    seed = 777 # 保持 seed 定義

    # Basic Training Params
    parser.add_argument("--env_name", type=str, default="ALE/Pong-v5", help="Gym environment ID")
    parser.add_argument("--save_dir", type=str, default="./results_rainbow_dqn", help="Directory to save results")
    parser.add_argument("--seed", type=int, default=seed, help="Random seed")
    parser.add_argument("--num_episodes", type=int, default=3000, help="Number of episodes to train")
    parser.add_argument("--max_episode_steps", type=int, default=27000, help="Max steps per episode before truncation (optional)")

    # Replay Buffer Params
    parser.add_argument("--memory_size", type=int, default=100000, help="Replay memory size")
    parser.add_argument("--replay_start_size", type=int, default=5000, help="Steps before starting training (aggressive)")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (aggressive)")

    # DQN Algorithm Params
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--n_step", type=int, default=3, help="N-step returns")
    parser.add_argument("--target_update", type=int, default=1500, help="Target network update frequency (in model updates, aggressive)")
    parser.add_argument("--learning_rate", type=float, default=1.5e-4, help="Initial learning rate for Adam (aggressive)")
    parser.add_argument("--lr_decay_steps", type=int, default=400000, help="Steps over which to decay LR (0 to disable)")
    parser.add_argument("--lr_end", type=float, default=1e-5, help="Final learning rate after decay")
    parser.add_argument("--gradient_clip", type=float, default=10.0, help="Gradient norm clipping value")

    # PER Params
    parser.add_argument("--alpha", type=float, default=0.5, help="PER alpha (prioritization exponent)")
    parser.add_argument("--beta", type=float, default=0.4, help="Initial PER beta (importance sampling exponent)")
    parser.add_argument("--beta_annealing_steps", type=int, default=400000, help="Steps to anneal beta to 1.0 (aggressive)")
    parser.add_argument("--prior_eps", type=float, default=1e-6, help="PER epsilon to avoid zero priorities")

    # Noisy Nets Params
    parser.add_argument("--noisy_std_init", type=float, default=0.5, help="Initial std dev for NoisyLinear")

    # Environment Params
    parser.add_argument("--frame_stack", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--frame_skip", type=int, default=4, help="Number of frames to skip per action")

    # Expert Data Params
    parser.add_argument("--expert_data_path", type=str, default="./expert_data_pong_100k.pkl", help="Path to expert 1-step data (.pkl)") # 請確認路徑
    parser.add_argument("--num_expert_transitions", type=int, default=10000, help="Number of expert transitions to pre-fill")

    # Misc Params
    parser.add_argument("--eval_episode_interval", type=int, default=25, help="Evaluate every N episodes")
    parser.add_argument("--compile", action='store_true', help="Enable torch.compile (experimental)")
    parser.add_argument("--wandb_project", type=str, default="DLP-Lab5-DQN-CartPole", help="WandB project name") # 建議改名 Pong
    parser.add_argument("--wandb_run_name", type=str, default=f"dqn_expert_fast_{time.strftime('%Y%m%d_%H%M')}")
    parser.add_argument("--train_frequency", type=int, default=4, help="Number of agent steps between model updates")
    parser.add_argument("--gradient_steps", type=int, default=1, help="Number of gradient steps per model update")

    # 解析參數 (保持不變)
    args = parser.parse_args()



    # --- Initialize WandB ---
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

    # --- Initialize and Train Agent ---
    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.train(num_episodes=args.num_episodes,
                eval_episode_interval=args.eval_episode_interval)

    wandb.finish()
    print("Run finished.")

# if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Fast Test Run") # 修改描述
    # seed = 777 # 保持固定種子

    # # ... (seed_torch, np.random.seed, etc.) ...

    # parser = argparse.ArgumentParser()

    # # --- **** 大幅縮減參數用於快速測試 **** ---
    # parser.add_argument("--num-episodes", type=int, default=2, help="Run only 2 episodes")
    # parser.add_argument("--memory-size", type=int, default=1000, help="Small memory")
    # parser.add_argument("--replay-start-size", type=int, default=100, help="Start training very early")
    # parser.add_argument("--batch-size", type=int, default=16, help="Small batch size")
    # parser.add_argument("--target-update", type=int, default=50, help="Frequent target update")
    # parser.add_argument("--max-episode-steps", type=int, default=50, help="Very short episodes")
    # parser.add_argument("--beta-annealing-steps", type=int, default=100, help="Very fast beta annealing")
    # parser.add_argument("--lr-decay-steps", type=int, default=0, help="Disable LR decay for quick test") # 設為 0 禁用
    # parser.add_argument("--eval-episode-interval", type=int, default=1, help="Evaluate every episode")
    # parser.add_argument("--num-expert-transitions", type=int, default=50, help="Load very few expert samples") # 減少專家數據
    # parser.add_argument("--train-frequency", type=int, default=1, help="Update every step for quick test") # 每步都更新
    # parser.add_argument("--gradient-steps", type=int, default=1, help="One gradient step")
    # # --- **** 縮減結束 **** ---

    # # --- 其他參數使用你的預設或保持不變 ---
    # parser.add_argument("--env-name", type=str, default="ALE/Pong-v5")
    # parser.add_argument("--save-dir", type=str, default="./results_quick_test") # 使用不同目錄
    # parser.add_argument("--seed", type=int, default=seed)
    # parser.add_argument("--gamma", type=float, default=0.99)
    # parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    # parser.add_argument("--lr-end", type=float, default=1e-5)
    # parser.add_argument("--gradient-clip", type=float, default=10.0)
    # parser.add_argument("--alpha", type=float, default=0.5)
    # parser.add_argument("--beta", type=float, default=0.4)
    # parser.add_argument("--prior-eps", type=float, default=1e-6)
    # parser.add_argument("--n-step", type=int, default=3)
    # parser.add_argument("--noisy-std-init", type=float, default=0.5)
    # parser.add_argument("--frame-stack", type=int, default=4)
    # parser.add_argument("--frame-skip", type=int, default=4)
    # parser.add_argument("--expert-data-path", type=str, default=None) # 可以設為你的測試數據路徑
    # parser.add_argument("--compile", action='store_true') # 保持可選
    # parser.add_argument("--wandb-project", type=str, default="DLP_Lab5_Quick_Test") # 使用不同項目名
    # parser.add_argument("--wandb-run-name", type=str, default=f"quick_test_{time.strftime('%H%M%S')}")

    # args = parser.parse_args()

    # # --- **** 禁用 WandB 和 Compile 以加速測試 **** ---
    # # wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args, mode="disabled") # 禁用 WandB
    # wandb.init(mode="disabled") # 更簡潔的禁用方式
    # args.compile = False # 強制禁用 Compile
    # # --- **** 禁用結束 **** ---


    # # --- Initialize and Train Agent ---
    # print("--- STARTING QUICK TEST RUN ---")
    # agent = DQNAgent(env_name=args.env_name, args=args)
    # # 移除 target_total_steps，因為 LR decay 被禁用了
    # agent.train(num_episodes=args.num_episodes,
    #             eval_episode_interval=args.eval_episode_interval)
    # print("--- QUICK TEST RUN FINISHED ---")

    # # wandb.finish() # 不需要 finish 如果 mode="disabled"