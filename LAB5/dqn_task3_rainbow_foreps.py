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
        self.acts_buf = np.zeros([size], dtype=np.uint8)
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
        act: np.ndarray, 
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

    
    def sample_batch_from_idxs(
        self, idxs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
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
        act: int, 
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

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
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
    def __init__(self, num_actions: int, noisy_std_init: float = 0.5):
        super(DQN, self).__init__()

        # Convolutional base (same as before)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Flatten: (N, 64*7*7 = 3136)
        self.feature_size = 64 * 7 * 7

        # set advantage layer (use NoisyLinear)
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.feature_size, 512, std_init=noisy_std_init),
            nn.ReLU(),
            NoisyLinear(512, num_actions, std_init=noisy_std_init) # Output: V(s)
        )
        # set value layer (use NoisyLinear)
        self.value_stream = nn.Sequential(
            NoisyLinear(self.feature_size, 512, std_init=noisy_std_init),
            nn.ReLU(),
            NoisyLinear(512, 1, std_init=noisy_std_init) # Output: V(s)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0 # Normalize input images
        feature = self.conv_layers(x)

        value = self.value_stream(feature)          # V(s)
        advantages = self.advantage_stream(feature) # A(s, a)

        # Combine value and advantages using Dueling formula:
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        """Resets noise in all NoisyLinear layers."""
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

        self.alpha = args.alpha
        self.beta = args.beta
        self.prior_eps = args.prior_eps
        
        self.gamma = args.gamma
        self.n_step = args.n_step
        self.memory_size = args.memory_size
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
        # networks: q_net, target_net
        self.noisy_std_init = args.noisy_std_init

        self.q_net = DQN(self.action_dim, noisy_std_init=self.noisy_std_init).to(self.device)
        self.q_net = torch.compile(self.q_net)  
        self.q_net.conv_layers.apply(init_weights)

        self.target_net = DQN(self.action_dim, noisy_std_init=self.noisy_std_init).to(self.device)
        self.target_net = torch.compile(self.target_net)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.beta_annealing_steps = args.beta_annealing_steps 
        # optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.learning_rate if hasattr(args, 'learning_rate') else 1e-4) # 使用 lr from args

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        self.is_test = False
        
        self.env_count = 0
        self.best_reward = -21 # Initialized for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.train_per_step = args.train_per_step
        self.early_stop_counter = 0 
        self.early_stop_threshold = 5 # Early stop threshold

        self.save_dir = os.path.join(
            args.save_dir,
            f"{time.strftime('%Y%m%d-%H%M%S')}_{args.wandb_run_name}"
            )
        os.makedirs(self.save_dir, exist_ok=True)


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
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.q_net.parameters(), 10.0)
        # self.optimizer_step() # 如果你之前 compile 了 optimizer.step
        self.optimizer.step() # 正常調用

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise for the ONLINE network only
        self.q_net.reset_noise()
        # self.target_net.reset_noise() # <--- 移除對 target_net 的 reset_noise

        return loss.item()
   
    def train(self, num_episodes: int, target_total_steps: int, eval_episode_interval: int):
        self.is_test = False
        obs_raw, _ = self.env.reset(seed=self.seed)
        state = self.preprocessor.reset(obs_raw)
        update_cnt = 0
        losses = []
        self.initial_beta = self.beta
        # 預先 compile optimizer step
        self.optimizer_step = torch.compile(self.optimizer.step)

        print(f"🚀 Starting fast training for {num_episodes} episodes...")

        for episode_idx in range(1, num_episodes + 1):
            obs_raw, _ = self.env.reset(seed=self.seed + episode_idx)
            state = self.preprocessor.reset(obs_raw)
            score, done, episode_steps = 0, False, 0

            while not done:
                
                self.q_net.train()
                # --- **** 添加結束 **** ---

                # **提速這邊** (選擇動作部分)
                # 使用 inference_mode 是為了不計算梯度，但對於 NoisyNet 的探索，
                # 我們其實需要網絡處於 train() 模式。select_action 內部調用 q_net 時，
                # 修正後的 NoisyLinear.forward 會因為 self.training 為 True 而添加噪聲。
                # 所以外層的 inference_mode 可能不是最合適的，但 select_action 內部用了 no_grad
                # select_action 內部的 no_grad 確保了選擇動作本身不計算梯度，是OK的。
                # 關鍵是 self.q_net 要處於 train() 模式。
                # with torch.inference_mode(): # 可以移除外層的 inference_mode
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device, dtype=torch.float32, non_blocking=True)
                action = self.select_action(state_tensor) # select_action 內部有 no_grad

                next_state, reward, done = self.env_step(action)
                score += reward

                transition_to_store = [state, action, reward, next_state, done]
                if self.use_n_step:
                    one_step_transition = self.memory_n.store(*transition_to_store)
                else:
                    one_step_transition = transition_to_store
                if one_step_transition:
                    self.memory.store(*one_step_transition)

                state = next_state
                self.env_count += 1
                episode_steps += 1

                # --- PER β退火 ---
                fraction = min(self.env_count / self.beta_annealing_steps, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

                # --- 更新網路 ---
                if len(self.memory) >= self.replay_start_size:
                    loss = self.update_model()
                    losses.append(loss)
                    update_cnt += 1

                    if update_cnt % 500 == 0 and wandb.run:
                        wandb.log({
                            "Episode": episode_idx,
                            "loss": loss,
                            "Env Step Count": self.env_count,
                            "Buffer Size": len(self.memory)
                        }, step=self.env_count)

                    if update_cnt % self.target_update == 0:
                        self._target_hard_update()

                if episode_steps >= self.max_episode_steps:
                    done = True
                # --- 定期保存 ---
                if self.env_count % 200000 == 0 and self.env_count <= target_total_steps:
                    model_path = os.path.join(self.save_dir, f"fast_lab5_pong_{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"[Checkpoint] Saved model at {model_path}")

                print(f"[FastTrain] Ep:{episode_idx}/{num_episodes} | Score:{score:.2f} | Steps:{episode_steps} | EnvS:{self.env_count}")

            # --- 評估 ---
            if len(self.memory) >= self.replay_start_size and episode_idx % eval_episode_interval == 0:
                eval_reward = self.evaluate()
                if eval_reward >= 19:
                    self.early_stop_counter += 1
                    if self.early_stop_counter >= self.early_stop_threshold:
                        print(f"[Early Stop] Reward >=19 for {self.early_stop_threshold} evaluations. Stopping!")
                        break
                else:
                    self.early_stop_counter = 0



        self.env.close()
        print("✨ Fast training finished.")


    def evaluate(self, n_episodes=30):
            """Evaluate the agent's performance."""
            print("Starting evaluation...")
            self.q_net.eval() # <--- 設置網絡為評估模式 (這會讓 NoisyLinear.forward 使用無噪聲路徑)
            # self.target_net.eval() # target_net 應該一直保持 eval 狀態，這裡調用也無妨

            # --- **** 移除以下區塊 **** ---
            # for module in self.q_net.modules():
            #     if isinstance(module, NoisyLinear):
            #         module.weight_epsilon.zero_() # 不再需要手動清零
            #         module.bias_epsilon.zero_()
            # --- **** 移除結束 **** ---

            total_rewards = []
            for ep in range(n_episodes):
                obs_raw, _ = self.test_env.reset(seed=self.seed + ep)
                state = self.preprocessor.reset(obs_raw)
                done = False
                episode_reward = 0
                ep_steps = 0

                while not done:
                    with torch.inference_mode(): # 使用 inference_mode 替代 no_grad，更現代
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device, non_blocking=True)
                        # select_action 內部會調用 q_net(state_tensor)
                        # 因為 q_net 已經是 eval 模式，這裡會使用無噪聲的確定性策略
                        action = self.select_action(state_tensor)

                    next_state, reward, done = self.test_env_step(action)
                    episode_reward += reward
                    state = next_state
                    ep_steps += 1
                    if ep_steps >= self.max_episode_steps:
                        print(f"Evaluation episode {ep+1} truncated at {ep_steps} steps.")
                        break

                total_rewards.append(episode_reward)
                # print(f"Evaluation episode {ep+1}/{n_episodes} finished. Reward: {episode_reward}") # 可以取消註釋以查看每回合分數

            avg_reward = np.mean(total_rewards)
            print(f"Evaluation finished. Average Reward over {n_episodes} episodes: {avg_reward:.2f}")

            # ... (保存模型和 wandb log 的邏輯不變) ...

            print(f"[Eval] EnvS: {self.env_count} Eval Reward: {avg_reward:.2f} EarlyStopCnt: {self.early_stop_counter}")

            if wandb.run:
                wandb.log({
                    "Step Count": self.env_count,
                    "Eval Reward": avg_reward,
                    "Best Eval Reward": self.best_reward
                }, step=self.env_count)

            self.q_net.train() # <--- 將網絡設置回訓練模式
            return avg_reward

    def _compute_dqn_loss(self,samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return q_net loss."""
        device = self.device  # for shortening the following lines
        # --- 從 uint8 轉為 FloatTensor (網絡內部會 / 255.0) ---
        state = torch.ByteTensor(samples["obs"]).to(device).float() 
        next_state = torch.ByteTensor(samples["next_obs"]).to(device).float()
        # ---
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)
        
        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.q_net(state).gather(1, action)
        # Double DQN: next action from online network, Q value from target network
        with torch.no_grad():
            next_action = self.q_net(next_state).argmax(1, keepdim=True)
            next_q = self.target_net(next_state).gather(1, next_action)
            target = reward + (1 - done) * gamma * next_q

        # calculate q_net loss
        elementwise_loss = F.mse_loss(curr_q_value, target, reduction='none')

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.target_net.load_state_dict(self.q_net.state_dict())
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    seed = 777

    def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    np.random.seed(seed)
    seed_torch(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # 啟用 TensorFloat32
    torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser()

    # 基本訓練參數
    parser.add_argument("--save-dir", type=str, default="./results_rainbow", help="Directory to save results")
    parser.add_argument("--memory-size", type=int, default=120000, help="Replay memory size")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--target-update", type=int, default=1500, help="Number of model updates between target network updates") # 基於 update_cnt
    parser.add_argument("--seed", type=int, default=seed, help="Random seed for reproducibility")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate for the optimizer")
    parser.add_argument("--num-episodes", type=int, default=2000, help="Number of episodes to train") # <--- 新增/替換
    parser.add_argument("--target-total-steps", type=int, default=2000000, help="Target total steps for beta annealing") # <--- 新增 (或重用 num_frames)
    parser.add_argument("--beta_annealing_steps", type=int, default=400000, help=" f") # <--- 新增 (或重用 num_frames)
    parser.add_argument("--eval-episode-interval", type=int, default=3, help="Evaluate every N episodes") # <--- 新增
    parser.add_argument("--max-episode-steps", type=int, default=5000, help="Max steps per episode before truncation")
    parser.add_argument("--replay-start-size", type=int, default=40000, help="Number of steps before starting training")
    parser.add_argument("--train-per-step", type=int, default=1, help="Train every env step (Keep if desired, though logic is slightly different now)") # 注意：現在是每步都嘗試訓練
    parser.add_argument("--noisy-std-init", type=float, default=0.5,
                        help="Initial standard deviation for NoisyLinear layers (e.g., 0.1, 0.4, 0.5)")

    # PER 相關參數
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter for prioritized experience replay") # 之前的 0.2 可能太小
    parser.add_argument("--beta", type=float, default=0.4, help="Initial beta parameter for prioritized experience replay") # 之前的 0.6 是結束值
    parser.add_argument("--prior-eps", type=float, default=1e-6, help="Epsilon added to priorities to avoid zero probability")

    # N-step Learning
    parser.add_argument("--n-step", type=int, default=3, help="Number of steps for multi-step returns")

    # WandB
    parser.add_argument("--wandb-run-name", type=str, default="pong-rainbow-episode-run")

    # Frame Skip (保持)
    parser.add_argument("--frame-skip", type=int, default=4) # 你可以從 args 控制 frame_skip

    args = parser.parse_args()
    wandb.init(project="DLP-Lab5-DQN-CartPole", name=args.wandb_run_name, save_code=True)

    # train
    agent = DQNAgent(args=args)
    agent.train(num_episodes=args.num_episodes,
                target_total_steps=args.target_total_steps,
                eval_episode_interval=args.eval_episode_interval)