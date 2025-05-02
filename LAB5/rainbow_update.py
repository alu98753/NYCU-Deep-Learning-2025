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
import math
import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import gymnasium as gym          # 已換成 gymnasium
import ale_py                    # 2025 spring 預設 atari backend
import cv2, numpy as np, random
from collections import deque
import torch, torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import random


# -*- coding: utf-8 -*-
"""Segment tree for Prioritized Replay Buffer."""

import operator
from typing import Callable
import time

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
    
class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        n_step: int = 1, 
        gamma: float = 0.99
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
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

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,
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
    
from tqdm import tqdm  # 放在檔案最上面
import collections

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
        obs_dim: int= None, 
        size: int = 100000 ,
        batch_size: int = 32, 
        alpha: float = 0.6,
        n_step: int = 1, 
        gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0
        self.max_size = size
        self.batch_size = batch_size
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = collections.deque(maxlen=n_step)
        self.size = 0
        self.ptr = 0

        # override the original flat obs_buf with image-shape buffer
        self.obs_buf = np.zeros((size, 4, 84, 84), dtype=np.uint8)
        self.next_obs_buf = np.zeros((size, 4, 84, 84), dtype=np.uint8)
        self.acts_buf = np.zeros([size], dtype=np.int64)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
class PrioritizedReplayBuffer: # 不再繼承 ReplayBuffer
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

        # Attributes for N-step Learning (Moved from ReplayBuffer)
        n_step_buffer (Deque): buffer for N-step transitions
        n_step (int): step number to calculate n-step td error
        gamma (float): discount factor
    """

    def __init__(
        self,
        obs_dim: Tuple[int, int, int], # 這裡應該是 image shape (C, H, W)
        size: int = 100000,
        batch_size: int = 32,
        alpha: float = 0.6,
        n_step: int = 3, # N-step Learning parameter
        gamma: float = 0.99, # Discount factor for N-step
    ):
        """Initialization."""
        assert alpha >= 0
        self.max_size = size
        self.batch_size = batch_size
        self.alpha = alpha

        # Buffers for storing n-step transitions
        # obs_dim should be something like (4, 84, 84) for stacked Atari frames
        c, h, w = obs_dim
        self.obs_buf = np.zeros([size, c, h, w], dtype=np.uint8) # Store uint8 images
        self.next_obs_buf = np.zeros([size, c, h, w], dtype=np.uint8) # Store uint8 images
        self.acts_buf = np.zeros([size], dtype=np.int64)
        self.rews_buf = np.zeros([size], dtype=np.float32) # These are N-step rewards
        self.done_buf = np.zeros(size, dtype=np.float32) # These are done flags for N-step transitions

        # For N-step Learning (moved from ReplayBuffer)
        self.n_step_buffer = collections.deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma # This gamma is for calculating n-step returns

        self.max_priority, self.ptr, self.size = 1.0, 0, 0

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
        self,
        obs: np.ndarray, # single step obs
        act: int, # single step action
        rew: float, # single step reward
        next_obs: np.ndarray, # single step next_obs
        done: bool, # single step done
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience in buffer (single step).
           If n-step transition is complete, store it and return.
        """
        # Store the single-step transition in the n-step buffer
        self.n_step_buffer.append((obs, act, rew, next_obs, done))

        # If n-step transition is not ready, just return
        if len(self.n_step_buffer) < self.n_step:
             # Return an empty tuple or similar to indicate no n-step transition was stored yet
             return ()

        # If n-step transition is ready, calculate it
        # The transition that starts the n-step sequence is the oldest one in the deque
        obs_n_step, act_n_step = self.n_step_buffer[0][:2]
        # Calculate the n-step reward, next_obs, and done
        rew_n_step, next_obs_n_step, done_n_step = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )

        # Store the calculated n-step transition into the main buffers
        self.obs_buf[self.ptr] = obs_n_step
        self.next_obs_buf[self.ptr] = next_obs_n_step
        self.acts_buf[self.ptr] = act_n_step
        self.rews_buf[self.ptr] = rew_n_step
        self.done_buf[self.ptr] = done_n_step

        # Update the segment trees with initial max priority
        self.sum_tree[self.ptr] = self.max_priority ** self.alpha
        self.min_tree[self.ptr] = self.max_priority ** self.alpha

        # Update buffer pointers and size
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        # Return the stored n-step transition (optional, useful for debugging/logging)
        # Or return an indicator that a transition was stored
        return (obs_n_step, act_n_step, rew_n_step, next_obs_n_step, done_n_step)


    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        # Retrieve stored n-step transitions from buffers
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices] # These are the n-step rewards
        done = self.done_buf[indices] # These are the n-step done flags

        # Calculate importance sampling weights
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews, # n-step rewards
            done=done, # n-step done flags
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            # Priority should only be updated for indices that are currently valid in the buffer
            # The check `0 <= idx < len(self)` is important
            assert 0 <= idx < self.size # Use self.size instead of self.__len__() to avoid recursion if len() uses size

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1) # Use self.__len__() or self.size
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            # Retrieve index from the sum tree based on upperbound
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min(0, len(self) - 1) / self.sum_tree.sum(0, len(self) - 1) # Use self.__len__() or self.size
        # Avoid division by zero if buffer is empty or sum is zero (though len(self) >= batch_size check should prevent this)
        max_weight = (p_min * len(self)) ** (-beta) if p_min > 1e-6 else 1.0 # Use self.__len__() or self.size

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum(0, len(self) - 1) # Use self.__len__() or self.size
        # Avoid division by zero
        weight = (p_sample * len(self)) ** (-beta) if p_sample > 1e-6 else 1.0 # Use self.__len__() or self.size
        weight = weight / max_weight

        return weight

    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done (from the perspective of the first state in the deque)."""
        # This function calculates the n-step return and the final state/done flag
        # based on the sequence of single-step transitions in the deque.

        # Start with the last transition's info
        # Reversed list(n_step_buffer) accesses transitions from newest to oldest
        rew, next_obs, done = n_step_buffer[-1][-3:] # last transition's reward, next_state, done

        # Iterate backwards from the second to last transition to the first
        # Accumulate discounted rewards
        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            # Accumulate reward: r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
            # If done at any intermediate step, the sum stops there.
            rew = r + gamma * rew * (1 - d) # (1-d) ensures accumulation stops if an intermediate done=True

            # The next_obs and done flag are from the *first* terminal state encountered,
            # or the final state if no terminal state is encountered within n steps.
            # This part in the original code looks slightly off. The final next_obs and done
            # should be from the transition *n steps away* from the start state,
            # *unless* a terminal state is reached earlier.
            # Let's re-check the original intent. The original code seems to mean:
            # If any transition in the sequence is done, the *final* done flag is True,
            # and the final next_obs is the next_obs of that terminal transition.
            # If no intermediate transition is done, the final done is the last transition's done,
            # and the final next_obs is the last transition's next_obs.
            # The original logic 'next_obs, done = (n_o, d) if d else (next_obs, done)' is correct
            # for finding the info of the *first* terminal state or the last state.
            next_obs, done = (n_o, d) if d else (next_obs, done)


        # The next_obs and done returned here are the next_obs and done *corresponding to the state reached after n steps*,
        # taking into account intermediate terminal states.
        return rew, next_obs, done


    def __len__(self) -> int:
        """Return the current size of the replay buffer."""
        return self.size # Return the number of stored n-step transitions 

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    
    
        
    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter
        
    """

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        std_init: float = 0.5,
    ):
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
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.
        
        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )
    
    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())

class RainbowCNN(nn.Module):
    def __init__(self, atom_size: int, action_dim: int, support: torch.Tensor):
        super().__init__()
        self.support = support
        self.action_dim, self.atom_size = action_dim, atom_size

        self.feature = nn.Sequential(          # ▸ 輸入 (4,84,84)
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(), # (32,20,20)
            nn.Conv2d(32,64,4,2), nn.ReLU(),   # (64,9,9)
            nn.Conv2d(64,64,3,1), nn.ReLU(),   # (64,7,7)
            nn.Flatten()
        )
        feat_dim = 64 * 7 * 7

        # dueling noisy heads
        self.adv_hid = NoisyLinear(feat_dim, 512)
        self.adv_out = NoisyLinear(512, action_dim * atom_size)

        self.val_hid = NoisyLinear(feat_dim, 512)
        self.val_out = NoisyLinear(512, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)
        
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature(x)  # ✅ 正確名稱：self.feature，不是 self.feature_layer

        adv_hid = F.relu(self.adv_hid(feature))     # ✅ 名稱修正
        val_hid = F.relu(self.val_hid(feature))     # ✅ 名稱修正

        advantage = self.adv_out(adv_hid).view(
            -1, self.action_dim, self.atom_size     # ✅ action_dim, atom_size 是正確的維度
        )
        value = self.val_out(val_hid).view(-1, 1, self.atom_size)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        dist = F.softmax(q_atoms, dim=-1).clamp(min=1e-3)  # avoid NaN

        return dist

    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.adv_hid.reset_noise()
        self.adv_out.reset_noise()
        self.val_hid.reset_noise()
        self.val_out.reset_noise()


class DQNAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor for TD target (single step gamma)
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done (single step)
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        n_step (int): step number to calculate n-step td error
        # memory_n removed
    """

    def __init__(
        self,
        env: gym.Env,
        memory_size: int,
        batch_size: int,
        target_update: int,
        seed: int,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: float = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0, # These will be overridden for Pong
        v_max: float = 200.0, # These will be overridden for Pong
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate (Note: lr is not a param here, using default Adam lr)
            gamma (float): discount factor (single step)
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Get environment details
        action_dim = env.action_space.n
        # Assuming the observation space is Box with shape (C, H, W)
        obs_shape = env.observation_space.shape
        if len(obs_shape) != 3:
             raise ValueError(f"Observation space shape expected to be (C, H, W), but got {obs_shape}")
        obs_dim = obs_shape

        # Categorical DQN parameters (Set specifically for Pong as per original code)
        self.v_min = -21
        self.v_max = 21
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # Networks
        self.dqn       = RainbowCNN(atom_size, action_dim, self.support).to(self.device)
        self.dqn_target= RainbowCNN(atom_size, action_dim, self.support).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval() # Set target network to eval mode

        # Environment and parameters
        self.env = env
        self.batch_size = batch_size
        self.target_update = target_update # in steps
        self.seed = seed
        self.gamma = gamma # single step gamma

        # PER
        self.beta = beta
        self.prior_eps = prior_eps
        # Use the modified PrioritizedReplayBuffer that handles n-step internally
        self.memory = PrioritizedReplayBuffer(
            obs_dim=obs_dim, # Pass the observation shape
            size=memory_size,
            batch_size=batch_size,
            alpha=alpha,
            n_step=n_step, # Pass n_step for the buffer to use
            gamma=gamma, # Pass gamma for the buffer to use in n-step return calculation
        )

        # N-step Learning parameter
        self.n_step = n_step # Store n_step in the agent as well

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=1e-4) # Added a default learning rate # 

        # transition to store in memory (single step transition)
        self.transition = list()

        # mode: train / test
        self.is_test = False

        # Tracking training progress
        self.update_cnt = 0
        
        self.losses = []
        self.scores = [] # Keep track of scores per episode
        self.score_frames = [] # Keep track of frame_idx when episode ends
        self.eval_scores = []
        self.eval_frames = []
        self.eval_interval = 50000 # Define how often to evaluate (in frames)

    def select_action(self, state: np.ndarray) -> int: # Action is an integer for discrete actions
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        # state shape is (C, H, W) for stacked frames
        # Need to convert np.uint8 to torch.float32 and normalize
        # Ensure state is in expected format (C, H, W), unsqueeze for batch dim (B, C, H, W)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.0


        # Set dqn to eval mode for action selection (important for Noisy Nets consistency)
        # Though reset_noise is called after update, eval mode ensures no new noise is generated here.
        if self.is_test:
            self.dqn.eval()
        else:
            self.dqn.train()

        with torch.no_grad():
            q_values = self.dqn(state_t)
            selected_action = q_values.argmax(1).item()

        return selected_action

    def step(self, action: int) -> Tuple[np.ndarray, np.float64, bool]: # action is int
        """Take an action and return the response of the env."""
        # env.step returns next_state (np.ndarray), reward (float), terminated (bool), truncated (bool), info (dict)
        next_state, reward, terminated, truncated, _ = self.env.step(action)

        # Clip reward for Atari
        reward = np.sign(reward).astype(np.float32)

        done = terminated or truncated

        if not self.is_test:
            # Complete the single-step transition
            self.transition += [reward, next_state, done]
            # Store the single-step transition in the buffer
            # The buffer will handle n-step accumulation and storage
            self.memory.store(*self.transition)

        return next_state, reward, done

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # Check if buffer has enough samples
        if len(self.memory) < self.batch_size:
            return 0.0 # Return 0 loss if not enough samples

        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        # samples now contain n-step transitions (obs, next_obs, acts, rews, done)

        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # Calculate the correct n-step gamma for the target calculation
        gamma_n_step = self.gamma ** self.n_step

        # Compute the categorical loss using the sampled n-step transitions
        # Pass the n-step samples and the correct gamma^n
        elementwise_loss = self._compute_dqn_loss(samples, gamma_n_step)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        # Use the elementwise loss from the n-step calculation
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise after the optimization step
        self.dqn.reset_noise()
        # Target network noise should also be reset if it uses Noisy layers
        self.dqn_target.reset_noise()

        self.update_cnt += 1 # Increment update counter

        # Perform hard update of target network
        if self.update_cnt % self.target_update == 0:
             self._target_hard_update()


        return loss.item()


    def train(self, num_frames: int, plotting_interval: int = 200):
        """Train the agent."""
        self.is_test = False

        # Reset the environment and get the initial state (stacked frames)
        state, _ = self.env.reset(seed=self.seed)
        # state should be a tuple of frames or a stacked numpy array depending on wrapper output
        # Assuming it's a stacked numpy array of shape (C, H, W)

        self.update_cnt = 0 # Reset update counter
        score = 0
        start_time = time.time()
        model_save_dir = "rainbow_models"
        os.makedirs(model_save_dir, exist_ok=True) # Create directory if it doesn't exist

        # Initialize tqdm progress bar
        with tqdm(range(1, num_frames + 1), desc="Training Progress") as pbar:
            for frame_idx in pbar: # frame_idx represents the number of steps taken in the environment
                action = self.select_action(state)
                # step returns next_state (stacked frames), reward (clipped), done (terminated or truncated)
                next_state, reward, done = self.step(action)

                # Update state for the next step
                state = next_state
                # Accumulate score (raw score, before clipping, if possible from env info, but using clipped here is fine for tracking)
                # Note: For real score tracking, you'd often get it from env.unwrapped.get_state() or info dict
                # Using the clipped reward is standard for the training signal, but the score logged should ideally be the true episodic score.
                # Let's stick to the clipped reward for simplicity as per the original code structure.
                score += reward

                # PER: increase beta linearly
                # Anneal beta from its initial value to 1.0 over the course of training
                # fraction is the training progress percentage (0.0 to 1.0)
                fraction = min(frame_idx / num_frames, 1.0)
                # Beta starts at self.beta (initial value) and increases towards 1.0
                self.beta = self.beta + fraction * (1.0 - self.beta) # Linear annealing

                # if training is ready (buffer has enough samples)
                # Note: The buffer stores n-step transitions.
                # We should probably wait longer than batch_size single steps before the first update.
                # A common practice is to wait until buffer size > batch_size * n_step or a fixed number of frames.
                # Let's keep the original len(self.memory) >= self.batch_size check for now,
                # but be aware that the first few updates might use incomplete n-step sequences
                # if the buffer implementation doesn't handle this edge case during sampling.
                # However, the `store` method only adds a transition when n-steps are complete,
                # so sampling will always provide complete n-step transitions once enough have been stored.
                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()
                    # Only append loss if update_model actually ran (returned non-zero)
                    if loss > 1e-9: # Check if loss is significant
                        self.losses.append(loss)
                        
                # --- Model Saving Logic ---

                # 1. Save model every 200,000 frames
                if frame_idx > 0 and frame_idx % 200000 == 0: # Ensure it's not just frame 0
                    save_path = os.path.join(model_save_dir, f"rainbow_dqn_frame_{frame_idx}.pth")
                    torch.save(self.dqn.state_dict(), save_path)
                    print(f"\nModel saved at frame {frame_idx} to {save_path}")

                # Perform evaluation periodically
                if frame_idx % self.eval_interval == 0:
                    avg_eval_score = self.evaluate()
                    self.eval_scores.append(avg_eval_score)
                    self.eval_frames.append(frame_idx)
                    # You might want to log eval score to wandb here
                    # wandb.log({"frame": frame_idx, "avg_eval_score": avg_eval_score})



                # if episode ends
                if done:
                    # Reset environment for a new episode
                    # Log the total score for the episode
                    self.scores.append(score)
                    self.score_frames.append(frame_idx) # Record frame_idx when episode ends
                    # NoisyNet: reset noise at the start of a new episode (optional but can help exploration)
                    # self.dqn.reset_noise()
                    # self.dqn_target.reset_noise()
                    # 2. Save model if episode score >= 18 (using the tracked score)
                    if score >= 18:
                        save_path = os.path.join(model_save_dir, f"rainbow_dqn_score_{score:.2f}_frame_{frame_idx}.pth")
                        torch.save(self.dqn.state_dict(), save_path)
                        print(f"\nModel saved at frame {frame_idx} due to episode score {score:.2f} >= 18 to {save_path}")

                    state, _ = self.env.reset(seed=self.seed)
                    score = 0 # Reset score for the next episode

                # Plotting training progress (current episode score, loss)
                # Maybe rename this plot function or create a separate one for eval scores
                # If plotting current episode score, keep the old _plot logic for the left subplot
                # If plotting eval scores, modify the left subplot to use eval_frames and eval_scores
                if frame_idx % plotting_interval == 0:
                    # Option 1: Plot current episode score (X-axis: Episode #) and Loss (X-axis: Update Step)
                    # This requires storing scores per episode index and loss per update index
                    # self._plot_train_progress(frame_idx, self.scores, self.losses) # Need to pass/store scores/losses

                    # Option 2: Plot Evaluation Score (X-axis: Frame #) and Loss (X-axis: Update Step)
                    self._plot(frame_idx) # Need a new plot function



                # Update tqdm postfix more frequently for better feedback
                if frame_idx % 1000 == 0: # Log every 1000 frames
                    avg_score_100 = np.mean(self.scores[-100:]) if self.scores else 0.0 # Avg score over last 100 episodes
                    avg_loss_1000 = np.mean(self.losses[-min(len(self.losses), 1000):]) if self.losses else 0.0 # Avg loss over last 1000 updates
                    pbar.set_postfix(
                        frames=frame_idx,
                        loss=f"{avg_loss_1000:.4f}",
                        score=f"{avg_score_100:.2f}",
                        mem=len(self.memory)
                    )

        # End of training
        print("\nTraining finished.")
        print(f"Total training time: {time.time() - start_time:.2f} seconds")
        self.env.close()


    def test(self, video_folder: str) -> None:
        """Test the agent."""
        self.is_test = True

        # for recording a video
        # Make a new environment instance for recording, don't wrap the training env directly
        # Also apply the same preprocessing and frame stacking
        # Note: RecordVideo should be one of the outer wrappers
        test_env = make_env(seed=self.seed, frame_stack=4) # Use the make_env helper
        test_env = gym.wrappers.RecordVideo(test_env, video_folder=video_folder, name_prefix="pong_test_")

        state, _ = test_env.reset(seed=self.seed)
        done = False
        score = 0

        print("Starting test...")
        # Set dqn to eval mode for testing
        self.dqn.eval()

        while not done:
            # Select action without storing transition
            # state_t shape is (1, C, H, W)
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.0
            # NoisyNet: in eval mode, mean is used (deterministic action)
            with torch.no_grad():
                action = self.dqn(state_t).argmax(1).item()

            # Step the test environment
            next_state, reward, terminated, truncated, _ = test_env.step(action)

            state = next_state
            # Accumulate true score (ideally, get from info if available, but using clipped here)
            score += reward # Using clipped reward for consistency with training score tracking
            done = terminated or truncated


        print(f"Test finished. Score: {score}")
        test_env.close()

        # No need to reset self.env = naive_env if we created a separate test env

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss.
           samples contain n-step transitions.
           gamma is gamma^n_step.
        """
        device = self.device  # for shortening the following lines
        # Sampled data (these are n-step transitions)
        # Convert uint8 observations to float and normalize
        state = torch.from_numpy(samples["obs"]).float().to(device) / 255.0
        next_state = torch.from_numpy(samples["next_obs"]).float().to(device) / 255.0
        action = torch.LongTensor(samples["acts"]).to(device)
        # rews and done are already n-step values from the buffer
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN: select action with policy network, evaluate with target network
            # Get Q-values (mean) from policy network for next state to select best action
            next_action = self.dqn(next_state).argmax(1)
            # Get distributions from target network for next state
            next_dist = self.dqn_target.dist(next_state)
            # Select the distributions corresponding to the chosen action
            next_dist = next_dist[range(self.batch_size), next_action] # Shape: (batch_size, atom_size)

            # Calculate the n-step target distribution projection (Tz)
            # t_z = R_n + gamma^n * Z(s_{t+n}, a*)
            # reward is R_n (n-step reward)
            # gamma is gamma^n (passed from update_model)
            # (1-done) ensures the target is just the reward if the n-step transition ends in a terminal state
            t_z = reward + (1 - done) * gamma * self.support # self.support shape: (atom_size,)
            # t_z shape: (batch_size, atom_size) - broadcasting reward/done/gamma to match support

            # Clamp the target values to the support range [v_min, v_max]
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)

            # Project Tz onto the support [v_min, v_max]
            # b = (Tz - v_min) / delta_z --> bin index
            b = (t_z - self.v_min) / delta_z # Shape: (batch_size, atom_size)
            # l and u are the lower and upper bin indices
            l = b.floor().long() # Shape: (batch_size, atom_size)
            u = b.ceil().long() # Should be ceil, not floor + 1, to handle exact bin values correctly
            # u = b.floor().long() + 1 # Original code had floor + 1, ceil is more standard for categorical projection

            # Ensure indices are within the valid range [0, atom_size - 1]
            l = l.clamp(0, self.atom_size - 1)
            u = u.clamp(0, self.atom_size - 1)


            # Distributional DP step - project target distribution
            # proj_dist[i][j] will store the probability mass from next_dist[i][k]
            # that falls into bin j when projected.
            proj_dist = torch.zeros(next_dist.size(), device=device) # Shape: (batch_size, atom_size)

            # Offset for indexing into the flattened proj_dist tensor
            offset = (
                torch.arange(self.batch_size, device=device).unsqueeze(1) * self.atom_size
            ) # Shape: (batch_size, 1)
            # Expand offset to match l and u shapes
            offset = offset.expand(self.batch_size, self.atom_size) # Shape: (batch_size, atom_size)

            # Scatter the probability mass
            # Probability mass at next_dist[i][k] is split between l[i][k] and u[i][k]
            # Mass for l[i][k] is next_dist[i][k] * (u[i][k] - b[i][k])
            # Mass for u[i][k] is next_dist[i][k] * (b[i][k] - l[i][k])

            # Add mass to lower bin l
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            # Add mass to upper bin u
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        # Calculate the policy distribution for the current state and action
        dist = self.dqn.dist(state) # Shape: (batch_size, action_dim, atom_size)
        # Select the distributions corresponding to the actions taken in the samples
        # action shape: (batch_size,)
        log_p = torch.log(dist[range(self.batch_size), action]) # Shape: (batch_size, atom_size)

        # Calculate the categorical cross-entropy loss
        # loss = -sum(proj_dist * log(policy_dist)) over atoms dimension
        elementwise_loss = -(proj_dist * log_p).sum(1) # Shape: (batch_size,)

        return elementwise_loss


    def _target_hard_update(self):
        """Hard update: target <- local."""
        print("Hard updating target network...")
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(
        self,
        frame_idx: int,
        # No need to pass scores and losses here anymore if storing internally
    ):
        """Plot Evaluation Score vs Frame and Loss vs Update Step."""
        save_dir = "plots"
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=(20, 5))

        # Plot Evaluation Scores vs Frame
        plt.subplot(1, 2, 1)
        if self.eval_frames and self.eval_scores:
            plt.plot(self.eval_frames, self.eval_scores, marker='o') # Use marker for eval points
            plt.title(f'frame {frame_idx}. Last Eval Avg Score: {self.eval_scores[-1]:.2f}')
            plt.xlabel("Frame")
            plt.ylabel("Average Evaluation Score")
            plt.grid(True)
        else:
            plt.title(f'frame {frame_idx}. Evaluation Scores: N/A')


        # Plot Losses vs Update Step (assuming self.losses is stored)
        plt.subplot(1, 2, 2)
        if hasattr(self, 'losses') and self.losses:
            rolling_window_loss = min(len(self.losses), 100)
            if rolling_window_loss > 0:
                rolling_mean_losses = np.convolve(self.losses, np.ones(rolling_window_loss)/rolling_window_loss, mode='valid')
                plt.plot(range(rolling_window_loss - 1, len(self.losses)), rolling_mean_losses)
                plt.title(f'Loss (Avg over last {rolling_window_loss}): {rolling_mean_losses[-1]:.4f}')
            else:
                plt.title('Loss: N/A')
        else:
            plt.title('Loss: No updates yet')

        plt.xlabel("Update Step")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.tight_layout()
        filename = os.path.join(save_dir, f"plot_frame_{frame_idx}.png")
        plt.savefig(filename)
        plt.close()

    def evaluate(self, num_episodes: int = 5) -> float:
        """Evaluate the agent's performance over several episodes."""
        print(f"\nStarting evaluation for {num_episodes} episodes...")
        self.is_test = True # Set to test mode
        self.dqn.eval() # Set network to eval mode (deterministic action)

        episode_scores = []
        for episode in range(num_episodes):
            state, _ = self.env.reset(seed=self.seed + episode) # Use different seeds for eval episodes
            done = False
            score = 0
            while not done:
                # Select action deterministically
                state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device) / 255.0
                with torch.no_grad():
                    action = self.dqn(state_t).argmax(1).item()

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                state = next_state
                score += np.sign(reward) # Use clipped reward for consistency if training uses it

                done = terminated or truncated

            episode_scores.append(score)
            print(f" - Episode {episode+1}: Score {score}")

        avg_score = np.mean(episode_scores)
        print(f"Evaluation finished. Average score: {avg_score:.2f}")

        self.is_test = False # Set back to train mode
        self.dqn.train() # Set network back to train mode
        # NoisyNet: reset noise for the start of training
        self.dqn.reset_noise()
        self.dqn_target.reset_noise() # Also reset target noise

        return avg_score

# environment setup (remains the same)
from gymnasium.wrappers import FrameStackObservation

def make_env(seed: int = 0, frame_stack: int = 4):
    env = gym.make(
        "ALE/Pong-v5",
        render_mode="rgb_array",            # 只回傳影像，不開視窗
        repeat_action_probability=0.0
    )
    env = gym.wrappers.AtariPreprocessing(
        env,
        grayscale_obs=True,
        scale_obs=False,    # 回傳 uint8 (0~255)
        screen_size=84,
        frame_skip=1
    )
    env = FrameStackObservation(env, stack_size=frame_stack)
    env.reset(seed=seed)
    return env



# seeding setup (remains the same)
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

import imageio.v2 as imageio

def quick_visualize_save_gif(agent, num_steps=300, gif_path="pong_preview.gif"):
    env = make_env(seed=0, frame_stack=4)
    state, _ = env.reset()
    frames = []

    for _ in range(num_steps):
        frames.append(env.render())       # 收集畫面

        with torch.no_grad():
            a = agent.dqn(
                torch.from_numpy(state).float().unsqueeze(0).to(agent.device)/255.0
            ).argmax(1).item()
        state, _, term, trunc, _ = env.step(a)
        if term or trunc:
            state, _ = env.reset()

    env.close()
    imageio.mimsave(gif_path, frames, fps=30)
    print(f"saved → {gif_path}")


# parameters and main execution block (remains largely the same, but use the modified Agent)
if __name__ == "__main__":
    seed = 777
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)

    # Parameters for Pong (you can adjust these)
    # num_frames = 3_000_000 # Total environment steps
    # memory_size = 1_000_000 # Replay buffer size (stores n-step transitions)
    # batch_size = 32
    # target_update = 8000 # Hard update target network every X *optimizer* steps
    # gamma = 0.99 # Single step discount factor
    # alpha = 0.5 # PER alpha
    # beta = 0.4 # PER beta (will anneal to 1.0)
    # v_min = -21 # C51 v_min for Pong
    # v_max = 21 # C51 v_max for Pong
    # atom_size = 51 # C51 number of atoms
    # n_step = 3 # N-step learning parameter

    # Reduced parameters for a quick test run
    num_frames = 10000000 # Reduced for faster testing
    memory_size = 50000 # Reduced buffer size
    batch_size = 32
    target_update = 1000 # Reduced target update frequency (in updates)
    gamma = 0.99
    alpha = 0.5
    beta = 0.4
    v_min = -21
    v_max = 21
    atom_size = 51
    n_step = 3 # Keep n-step same

    # Create environment
    env = make_env(seed)
    obs_shape = env.observation_space.shape # Get observation shape from the wrapped env

    # Create agent with the modified buffer
    agent = DQNAgent(
        env,
        memory_size = memory_size,
        batch_size  = batch_size,
        target_update = target_update,
        seed = seed,
        gamma = gamma,
        alpha = alpha,
        beta  = beta,
        v_min = v_min,
        v_max = v_max,
        atom_size = atom_size,
        n_step = n_step,
    )

    print(f"Agent initialized on device: {agent.device}")
    print(f"Observation shape: {obs_shape}")
    print(f"Action space size: {env.action_space.n}")
    print(f"Using n-step: {n_step}")
    print(f"Using gamma: {gamma}")
    print(f"Using gamma^n: {gamma**n_step:.4f}")
    print(f"Buffer size: {memory_size}")
    print(f"Batch size: {batch_size}")
    print(f"Target update frequency (updates): {target_update}")
    print(f"PER alpha: {alpha}")
    print(f"PER beta (initial): {beta}")
    print(f"C51 v_min: {v_min}, v_max: {v_max}, atom_size: {atom_size}")

    # 🛠 訓練前先 quick visualize 看一看
    quick_visualize_save_gif(agent, num_steps=300)    # Train the agent
    # plotting_interval is in frames
    agent.train(num_frames = num_frames, plotting_interval = 10000) # Plot every 10000 frames

    # Optional: Save the trained model
    # model_save_path = "rainbow_dqn_pong.pth"
    # torch.save(agent.dqn.state_dict(), model_save_path)
    # print(f"Model saved to {model_save_path}")

    # Optional: Test the agent after training
    # video_folder = "pong_test_videos"
    # agent.test(video_folder=video_folder)