# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL - Rainbow DQN Adaptation
# Original Rainbow Code Contributor: (Your Name/ID if desired)
# Style Adaptation Contributor: Gemini
from torch.nn.utils import clip_grad_norm_
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy # Renamed from 'numpy as np' to match sample SumTree usage
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque # Keep deque for n-step buffer in Agent
import wandb # Added wandb
import argparse # Added argparse
import time
import math
# Removed operator import as sample SumTree doesn't use it explicitly
from typing import Deque, Dict, List, Tuple # Keep necessary typing for Agent later

gym.register_envs(ale_py) # Keep environment registration

# Helper function from Sample Code for weight initialization
def init_weights(m):
    """Initialize weights using Kaiming uniform for Conv/Linear layers, but skip NoisyLinear."""
    if isinstance(m, nn.Conv2d) or (isinstance(m, nn.Linear) and not isinstance(m, NoisyLinear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Atari Preprocessor from Sample Code
class AtariPreprocessor:
    """Preprocessing the state input for Atari environments."""
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        """Convert to grayscale and resize."""
        # Input obs shape: (H, W, C) or similar RGB format
        if len(obs.shape) == 1:  # Handle potential 1D states (though unlikely for Atari)
            return obs
        else:
            # Ensure input is uint8 for cvtColor if it's float
            if obs.dtype != numpy.uint8:
                 # Assuming float is 0-255 range, adjust if it's 0-1
                 obs = obs.astype(numpy.uint8)
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            return resized # Returns shape (84, 84)

    def reset(self, obs):
        """Reset frame buffer with the first observation."""
        frame = self.preprocess(obs)
        if len(frame.shape) == 1:
             return frame # Handle non-image case
        # Initialize deque with the first frame repeated frame_stack times
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        # Stack frames along the first dimension (channel dim for PyTorch Conv2d)
        return numpy.stack(self.frames, axis=0) # Returns shape (frame_stack, 84, 84)

    def step(self, obs):
        """Process a new observation and add to frame buffer."""
        frame = self.preprocess(obs)
        if len(frame.shape) == 1:
             return frame # Handle non-image case
        self.frames.append(frame)
        # Stack frames along the first dimension
        return numpy.stack(self.frames, axis=0) # Returns shape (frame_stack, 84, 84)


# SumTree from Sample Code (Using numpy)
class SumTree:
    """SumTree data structure for Prioritized Replay Buffer."""
    write = 0 # Tracks the next position to write data

    def __init__(self, capacity):
        self.capacity = capacity
        # Tree structure: Stores priorities. Size is 2*capacity - 1.
        # Leaves start at index capacity - 1.
        self.tree = numpy.zeros(2 * capacity - 1)
        # Data storage: Stores transitions. Size is capacity.
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0 # Number of entries currently in the buffer

    def _propagate(self, idx, change):
        """Propagate priority changes up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find sample index based on cumulative priority 's'."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree): # Reached a leaf node
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Get the total priority (root node value)."""
        return self.tree[0]

    def add(self, p, data):
        """Store priority p and sample data."""
        idx = self.write + self.capacity - 1 # Index in the tree array for the leaf

        self.data[self.write] = data # Store data in the data array
        self.update(idx, p) # Update the tree with the new priority

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0 # Wrap around if capacity is reached

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """Update priority of a node and propagate the change."""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """Get leaf index, priority value, and data for a sample value 's'."""
        idx = self._retrieve(0, s) # Tree index of the leaf
        dataIdx = idx - self.capacity + 1 # Corresponding index in the data array
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    # 修改 __init__ 簽名以接收 beta_start 和 total_train_steps
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, total_train_steps=1000000, reward_scale=1.0, epsilon=1e-8):
        if reward_scale <= 0:
            print(f"Warning: reward_scale must be positive, received {reward_scale}. Setting to 1.0.")
            reward_scale = 1.0

        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start # 使用傳入的初始 beta
        self.beta_final = 1.0 # 目標 beta 通常是 1.0
        self.epsilon = epsilon
        self.reward_scale = reward_scale

        # 基於總訓練步數計算 beta 每步的增量
        if total_train_steps > 0:
             # 計算從 beta_start 到 beta_final 所需的每步增量
             self.beta_increment_per_sampling = (self.beta_final - self.beta) / total_train_steps
        else:
             self.beta_increment_per_sampling = 0 # 如果總步數為0則不增加

        print(f"PER Beta annealing: start={self.beta}, final={self.beta_final}, increment={self.beta_increment_per_sampling:.8e} per step over {total_train_steps} steps")


    def __len__(self):
        """Return the current number of transitions in the buffer."""
        return self.tree.n_entries

    def add(self, transition, error):
         """Add a new n-step transition and its initial TD error.

         Args:
             transition (tuple): The n-step transition (S_start, A_start, R_n_step, S_end, D_end).
             error (float): The TD error calculated for this n-step transition.
         """
         # Calculate priority based on scaled error
         scaled_error = abs(error / self.reward_scale)
         priority = (scaled_error + self.epsilon) ** self.alpha
         # Ensure priority is not zero, though epsilon should handle this
         priority = max(priority, self.epsilon) # Add explicit safeguard
         self.tree.add(priority, transition)

    def sample(self, batch_size):
        """Sample a batch of transitions with priorities and IS weights.

        Returns:
            tuple: A tuple containing:
                - states (numpy.ndarray): Batch of initial states (S_start from n-step).
                - actions (torch.Tensor): Batch of actions (A_start from n-step).
                - rewards (torch.Tensor): Batch of n-step returns (R_n_step).
                - next_states (numpy.ndarray): Batch of final states (S_end from n-step).
                - dones (torch.Tensor): Batch of done flags (D_end from n-step).
                - weights (torch.Tensor): Importance sampling weights.
                - indices (list): List of tree indices for the sampled transitions.
            Or None if sampling fails.
        """
        # Anneal beta towards beta_final
        if self.beta < self.beta_final: # 只有當 beta 未達到最終值時才增加
            self.beta += self.beta_increment_per_sampling
            # 確保不會超過最終值 (雖然線性增加理論上不會, 但加上更保險)
            self.beta = min(self.beta, self.beta_final)
        batch_data = []
        idxs = []
        priorities = []
        
        # Ensure tree has entries before proceeding
        if self.tree.n_entries == 0:
            print("Error: Attempting to sample from an empty buffer.")
            return None

        segment = self.tree.total() / batch_size

        valid_samples_found = 0
        attempts = 0
        max_attempts = batch_size * 5 # Limit attempts

        while valid_samples_found < batch_size and attempts < max_attempts:
            attempts += 1
            i = valid_samples_found # Use current count for segment calculation
            a = segment * i
            b = segment * (i + 1)
            # Ensure s does not exceed total priority, especially when buffer is not full
            # Add small epsilon to upper bound to handle potential float precision issues with total()
            s = random.uniform(a, min(b, self.tree.total() + 1e-9))
            s = min(s, self.tree.total()) # Ensure s is not greater than total

            try:
                idx, priority, data = self.tree.get(s)

                # --- Check retrieved data type ---
                # Expecting a tuple of length 5: (S, A, R_n, S_next_n, D_n)
                if isinstance(data, tuple) and len(data) == 5:
                    batch_data.append(data)
                    idxs.append(idx)
                    priorities.append(priority)
                    valid_samples_found += 1
                else:
                    # Log unexpected data type - likely the initial 0
                    # This might happen if sampling hits an uninitialized slot
                    # print(f"Warning: Sampled unexpected data type ({type(data)}) at tree index {idx} (priority {priority:.4f}, s {s:.4f}, total {self.tree.total():.4f}). Retrying sample.")
                    # Simply continue to the next attempt without incrementing valid_samples_found
                    continue # Retry sampling different s

            except AssertionError as e:
                print(f"Warning: SumTree get failed for s={s}, total={self.tree.total()}. Error: {e}. Retrying sample.")
                continue # Retry sampling
            except Exception as e: # Catch other potential errors during get
                print(f"Error during SumTree.get: {e}. Retrying sample.")
                continue # Retry sampling

        # After the loop, check if enough samples were collected
        if valid_samples_found < batch_size:
            print(f"Error: Could only sample {valid_samples_found}/{batch_size} valid transitions after {max_attempts} attempts. Check buffer state or sampling logic.")
            # Returning None as the batch is incomplete
            return None

        # --- Unpacking and Conversion (if batch_size samples were found) ---
        try:
            sampling_probabilities = numpy.array(priorities) / self.tree.total()

            weights = numpy.power(self.tree.n_entries * sampling_probabilities, -self.beta)
            if weights.max() > 1e-9:
                weights /= weights.max()
            else:
                weights = numpy.ones_like(weights)

            # This is the line that caused the original error
            # It should work now if batch_data only contains tuples
            batch = list(zip(*batch_data))

            states = numpy.array(batch[0])
            actions = torch.tensor(batch[1], dtype=torch.int64)
            rewards = torch.tensor(batch[2], dtype=torch.float32)
            next_states = numpy.array(batch[3])
            dones = torch.tensor(batch[4], dtype=torch.float32)
            weights = torch.tensor(weights, dtype=torch.float32)

            return states, actions, rewards, next_states, dones, weights, idxs

        except Exception as e: # Catch potential errors during unpacking/conversion
            print(f"Error during batch unpacking or conversion: {e}")
            print(f"Batch data length: {len(batch_data)}, Priorities length: {len(priorities)}")
            # Optionally print more details about batch_data structure if error persists
            return None

    def update_priorities(self, indices, errors):
         """Update priorities of sampled transitions based on new TD errors.

         Args:
             indices (list or numpy.ndarray): List of tree indices to update.
             errors (list or numpy.ndarray or torch.Tensor): New absolute TD errors for the samples.
         """
         if isinstance(errors, torch.Tensor):
             errors = errors.abs().detach().cpu().numpy() # Ensure numpy array of positive errors

         for i, idx in enumerate(indices):
             scaled_error = abs(errors[i] / self.reward_scale)
             priority = (scaled_error + self.epsilon) ** self.alpha
             # Ensure priority is not zero
             priority = max(priority, self.epsilon)
             # Check index validity before updating (should be valid if sampled correctly)
             if idx < (self.tree.capacity -1 + self.tree.capacity): # Check if idx is within tree bounds
                self.tree.update(idx, priority)
             else:
                print(f"Warning: Invalid index ({idx}) received in update_priorities. Max tree index: {2*self.tree.capacity - 2}. Skipping update for this index.")
                
# Keep NoisyLinear class definition as provided by the user
class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.
    (Implementation as provided in the user's rainbow.py)
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
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise: sign(randn) * sqrt(|randn|)."""
        # Create noise on the same device as the parameters
        x = torch.randn(size, device=self.weight_mu.device) 
        return x.sign().mul(x.abs().sqrt())


# Refactored Network Class (Style Aligned with Sample's DQN)
class DQN(nn.Module):
    """ Deep Q-Network with switchable components (Dueling, Noisy, C51).
    """
    def __init__(self, num_actions: int, frame_stack: int = 4,
                 # Flags to control components
                 use_dueling: bool = False,
                 use_noisy: bool = False,
                 use_distributional: bool = False,
                 # C51 params (only used if use_distributional)
                 atom_size: int = 51, v_min: float = -10.0, v_max: float = 10.0):
        """Initialization."""
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.frame_stack = frame_stack
        self.use_dueling = use_dueling
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional
        self.atom_size = atom_size if use_distributional else 1 # Set atom_size=1 if not distributional
        self.v_min = v_min
        self.v_max = v_max

        # Define Linear layer type based on whether Noisy Nets are used
        linear_layer = NoisyLinear if use_noisy else nn.Linear

        # Define support only if distributional
        if use_distributional:
            self.register_buffer(
                "support",
                torch.linspace(self.v_min, self.v_max, self.atom_size)
            )
        else:
            # If not distributional, support is not needed for Q-value calculation
            self.support = None # Or torch.zeros(1) if needed as placeholder

        # --- CNN Feature Extractor (remains the same) ---
        self.feature_layer = nn.Sequential(
            nn.Conv2d(self.frame_stack, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.feature_dim = 3136

        # --- Define Final Layers based on Flags ---
        if self.use_dueling:
            # Dueling Architecture
            # Advantage Stream
            self.advantage_hidden_layer = linear_layer(self.feature_dim, 512)
            # Output size depends on distributional or not
            adv_out_size = num_actions * self.atom_size
            self.advantage_layer = linear_layer(512, adv_out_size)

            # Value Stream
            self.value_hidden_layer = linear_layer(self.feature_dim, 512)
            # Output size depends on distributional or not
            val_out_size = self.atom_size # Only 1 set of atoms for value
            self.value_layer = linear_layer(512, val_out_size)
        else:
            # Standard (non-dueling) Architecture
            self.common_hidden_layer = linear_layer(self.feature_dim, 512)
            # Output size depends on distributional or not
            final_out_size = num_actions * self.atom_size
            self.final_layer = linear_layer(512, final_out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Calculates Q-values or distribution logits based on flags.
            Returns expected Q-values regardless.
        """
        x = x / 255.0
        feature = self.feature_layer(x)

        if self.use_dueling:
            adv_hid = F.relu(self.advantage_hidden_layer(feature))
            val_hid = F.relu(self.value_hidden_layer(feature))
            # Raw outputs (logits for C51, Q-values otherwise)
            advantage = self.advantage_layer(adv_hid)
            value = self.value_layer(val_hid)

            # Reshape for distributional or standard Q
            advantage = advantage.view(-1, self.num_actions, self.atom_size)
            value = value.view(-1, 1, self.atom_size)

            # Combine streams: Q = V + (A - mean(A))
            q_outputs = value + advantage - advantage.mean(dim=1, keepdim=True)

        else: # Non-Dueling
            common_hid = F.relu(self.common_hidden_layer(feature))
            q_outputs = self.final_layer(common_hid)
            # Reshape for distributional or standard Q
            q_outputs = q_outputs.view(-1, self.num_actions, self.atom_size)

        # If distributional, apply softmax and calculate expected Q-value
        if self.use_distributional:
            dist = F.softmax(q_outputs, dim=-1).clamp(min=1e-3)
            q_values = torch.sum(dist * self.support, dim=2)
        else: # Not distributional
            # q_outputs are already Q-values (since atom_size=1)
            # Squeeze the last dimension
            q_values = q_outputs.squeeze(-1) # Shape: (N, num_actions)

        return q_values

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """ Calculates the probability distribution P(z | s, a) for each action.
            Only makes sense if use_distributional is True.
            Assumes input x is already normalized.
        """
        if not self.use_distributional:
            raise RuntimeError("dist() method called when not using distributional RL")

        feature = self.feature_layer(x) # Input x is assumed normalized here

        if self.use_dueling:
            adv_hid = F.relu(self.advantage_hidden_layer(feature))
            val_hid = F.relu(self.value_hidden_layer(feature))
            advantage_logits = self.advantage_layer(adv_hid).view(-1, self.num_actions, self.atom_size)
            value_logits = self.value_layer(val_hid).view(-1, 1, self.atom_size)
            q_logits = value_logits + advantage_logits - advantage_logits.mean(dim=1, keepdim=True)
        else: # Non-Dueling
            common_hid = F.relu(self.common_hidden_layer(feature))
            q_logits = self.final_layer(common_hid).view(-1, self.num_actions, self.atom_size)

        dist = F.softmax(q_logits, dim=-1).clamp(min=1e-3)
        return dist

    def reset_noise(self):
        """Reset noise only if Noisy Nets are used."""
        if self.use_noisy:
            # Iterate through layers and call reset_noise if it's NoisyLinear
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()     

# (Previous code: Imports, init_weights, AtariPreprocessor, SumTree, PrioritizedReplayBuffer, NoisyLinear, DQN)

# Environment creation helper (modified)
def make_env(env_name: str, seed: int = None, render_mode: str = "rgb_array"):
    """Creates the base Atari environment.
       Preprocessing and FrameStack are handled by the agent's preprocessor.
    """
    # Note: repeat_action_probability=0.0 might differ from defaults.
    env = gym.make(env_name, render_mode=render_mode,frameskip=4, repeat_action_probability=0.0)
    # Seeding is usually handled by wrappers or reset, but setting here doesn't hurt
    if seed is not None:
       env.reset(seed=seed) # Seed on reset is preferred
    return env

# Seeding helper (from user code)
def seed_torch(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available(): # Check if CUDA is available
        torch.cuda.manual_seed(seed)
        # Optional: These might slow down training but improve reproducibility
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
    numpy.random.seed(seed)
    random.seed(seed)

class PongRewardShapingWrapper(gym.Wrapper):
    """ Applies reward shaping and logs the original reward in the info dict.
    """
    def __init__(self, env, rally_reward=0.01): # Using 0.01 as requested
        super().__init__(env)
        self.rally_reward = rally_reward
        self.current_original_reward = 0 # Store last original reward if needed
        print(f"PongRewardShapingWrapper initialized with rally_reward={rally_reward}")

    def step(self, action):
        """ Modifies the reward and adds original reward to info dict."""
        obs, original_reward, terminated, truncated, info = self.env.step(action)
        self.current_original_reward = original_reward # Store it

        # Apply shaping
        if original_reward == 0:
            shaped_reward = self.rally_reward
        else:
            shaped_reward = original_reward # Keep original +1/-1

        # Add original reward to info dict
        if info is None: info = {} # Ensure info dict exists
        info['original_reward'] = original_reward

        return obs, shaped_reward, terminated, truncated, info , original_reward

    # You might need reset if you track episode-level original score internally
    # def reset(self, **kwargs):
    #     obs, info = self.env.reset(**kwargs)
    #     self.current_original_reward = 0
    #     if info is None: info = {}
    #     info['original_reward'] = 0 # Reset original reward info
    #     return obs, info


# Refactored DQNAgent Class (Aligned with Sample Style)
class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        """Initialization."""
        if args is None:
            raise ValueError("Agent requires arguments (args) from argparse.")

        self.args = args
        self.env_name = env_name
        self.seed = args.seed

        # Store component flags
        self.use_dueling = args.use_dueling
        self.use_noisy = args.use_noisy
        self.use_distributional = args.use_distributional

        # Setup device, environment, preprocessor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        # Create base environments
        base_env = make_env(env_name, seed=self.seed)
        self.test_env = make_env(env_name, seed=self.seed + 1)

        # --- Apply Reward Shaping Wrapper --- # <--- NEW
        self.env = PongRewardShapingWrapper(base_env, rally_reward=args.rally_reward)
        print(f"Applied PongRewardShapingWrapper with rally_reward={args.rally_reward} to TRAINING env.")
        # Note: Evaluating with shaping might give inflated scores vs original Pong.
        # For official evaluation, you might want to use the base_test_env without the wrapper.
        # Let's keep it wrapped for now to see if the agent learns the shaped objective.

        print("Applied PongRewardShapingWrapper to env and test_env.")

        self.num_actions = self.env.action_space.n # Get action space from wrapped env
        self.preprocessor = AtariPreprocessor(frame_stack=args.frame_stack)

        # Store distributional parameters (needed regardless for DQN init signature, but only used if distributional)
        self.v_min = args.v_min
        self.v_max = args.v_max
        self.atom_size = args.atom_size

        # --- Epsilon-Greedy Setup (if not using Noisy Nets) ---
        if not self.use_noisy:
            self.epsilon = args.epsilon_start
            self.epsilon_final = args.epsilon_final
            self.epsilon_decay_rate = args.epsilon_decay_rate # Calculated in __main__
            print(f"Using Epsilon-Greedy: Start={self.epsilon}, Final={self.epsilon_final}, Decay Rate={self.epsilon_decay_rate}")
        else:
            self.epsilon = 0 # Epsilon not used with Noisy Nets

        # --- Setup Networks ---
        # Pass component flags to DQN constructor
        self.q_net = DQN(
            num_actions=self.num_actions, frame_stack=args.frame_stack,
            use_dueling=self.use_dueling, use_noisy=self.use_noisy, use_distributional=self.use_distributional,
            atom_size=self.atom_size, v_min=self.v_min, v_max=self.v_max
        ).to(self.device)
        self.target_net = DQN(
            num_actions=self.num_actions, frame_stack=args.frame_stack,
            use_dueling=self.use_dueling, use_noisy=self.use_noisy, use_distributional=self.use_distributional,
            atom_size=self.atom_size, v_min=self.v_min, v_max=self.v_max
        ).to(self.device)

        self.q_net.apply(init_weights) # init_weights now skips NoisyLinear
        self.target_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # --- Setup Optimizer ---
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # --- Setup Replay Buffer (PER enabled) ---
        self.memory = PrioritizedReplayBuffer(
            capacity=args.memory_size, alpha=args.alpha, beta_start=args.beta,
            total_train_steps=args.total_train_steps, reward_scale=args.reward_scale
        )

        # --- N-step Buffer ---
        self.n_step = args.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Store other hyperparameters
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.target_update_frequency = args.target_update_frequency
        self.replay_start_size = args.replay_start_size
        self.max_episode_steps = args.max_episode_steps
        self.train_per_step = args.train_per_step

        # Counters and tracking
        self.env_count = 0
        self.train_count = 0
        self.best_reward = -float('inf')

        # Save directory setup
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        run_name = args.wandb_run_name if args.wandb_run_name else f"dqn_variant_{self.env_name.split('/')[-1]}_{timestamp}"
        self.save_dir = os.path.join(args.save_dir, run_name)
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"Results and models will be saved in: {self.save_dir}")
        print(f"Config: Dueling={self.use_dueling}, Noisy={self.use_noisy}, Distributional={self.use_distributional}, N-step={self.n_step}, PER=True")

    def select_action(self, state: numpy.ndarray) -> int:
        """Select action using Q-network. Uses Noisy Nets if enabled, otherwise epsilon-greedy."""
        if not self.use_noisy and random.random() < self.epsilon:
            # Epsilon-greedy exploration
            selected_action = random.randrange(self.num_actions)
        else:
            # Greedy action based on Q-values (Noisy Nets exploration is internal)
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.q_net.eval() # Set to eval mode for deterministic action selection (esp. for Noisy mean)
            with torch.no_grad():
                q_values = self.q_net(state_t) # DQN.forward returns expected Q-values
                selected_action = q_values.argmax(1).item()
            self.q_net.train() # Set back to train mode
        return selected_action

    # _calculate_n_step_return remains the same
    def _calculate_n_step_return(self, current_n_step_buffer):
         # ... (keep implementation)
         R = 0
         S, A = current_n_step_buffer[0][:2] # Initial state and action
         D = False
         final_next_state = current_n_step_buffer[-1][3] # The state after n steps
         for i, (_, _, r, _, done_flag) in enumerate(current_n_step_buffer):
             R += (self.gamma ** i) * r
             if done_flag:
                 D = True # If any step was terminal, the n-step transition is considered terminal
                 break # Stop accumulating reward based on intermediate done
         S_next = final_next_state
         return S, A, R, S_next, D

    # _calculate_initial_td_error not needed if adding with max priority

    def run(self, episodes):
        """Main training loop over episodes."""
        total_start_time = time.time()
        for ep in range(episodes):
            # ... (episode initialization: reset env, state, buffer, rewards, steps, done) ...
            episode_start_time = time.time()
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            self.n_step_buffer.clear()
            original_episode_reward = 0
            episode_steps = 0
            done = False

            # --- Track both rewards per episode ---
            accumulated_shaped_reward = 0
            accumulated_original_reward = 0
            # ---

            while not done and episode_steps < self.max_episode_steps:
                # --- Epsilon Decay ---
                if not self.use_noisy and self.env_count >= self.replay_start_size:
                    self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_decay_rate)

                action = self.select_action(state)
                next_obs, shaped_reward, terminated, truncated, _, original_reward = self.env.step(action)
                done = terminated or truncated

                accumulated_shaped_reward += shaped_reward
                accumulated_original_reward += original_reward
                
                next_state = self.preprocessor.step(next_obs)
                self.n_step_buffer.append((state, action, shaped_reward, next_state, done))

                if len(self.n_step_buffer) == self.n_step:
                    S, A, R_n, S_next_n, D_n = self._calculate_n_step_return(self.n_step_buffer)
                    # Add with max priority
                    max_error_proxy = self.memory.reward_scale
                    self.memory.add((S, A, R_n, S_next_n, D_n), max_error_proxy)

                state = next_state
                self.env_count += 1
                episode_steps += 1

                if self.env_count >= self.replay_start_size:
                    for _ in range(self.train_per_step):
                        self.train()
                # --- Logging and Saving (Periodic) ---
                if self.env_count % 1000 == 0:
                    log_dict = {
                        "Progress/Env Steps": self.env_count,
                        "Progress/Train Steps": self.train_count,
                        "Parameters/PER Beta": self.memory.beta,
                    }
                    if not self.use_noisy:
                         log_dict["Parameters/Epsilon"] = self.epsilon
                    print(f"[Progress] Env Steps: {self.env_count}, Train Steps: {self.train_count}, Beta: {self.memory.beta:.4f}" + (f", Epsilon: {self.epsilon:.4f}" if not self.use_noisy else ""))
                    wandb.log(log_dict, step=self.env_count)

                save_interval = getattr(self.args, "save_interval", 200000)
                if self.env_count > 0 and self.env_count % save_interval == 0:
                    # ... (save snapshot) ...
                    snapshot_path = os.path.join(self.save_dir, f"q_net_snapshot_{self.env_count}.pt")
                    torch.save(self.q_net.state_dict(), snapshot_path)
                    print(f"Saved snapshot to {snapshot_path}")

            # --- End of Episode ---
            episode_duration = time.time() - episode_start_time

            # --- Flush n-step buffer ---
            while len(self.n_step_buffer) > 0:
                S, A, R_n, S_next_n, D_n = self._calculate_n_step_return(self.n_step_buffer)
                max_error_proxy = self.memory.reward_scale
                self.memory.add((S, A, R_n, S_next_n, D_n), max_error_proxy)
                self.n_step_buffer.popleft()

            # --- Log Episode Results (Log BOTH rewards) ---
            print(f"[Episode End] Ep: {ep+1}/{episodes}, Shaped Reward: {accumulated_shaped_reward:.2f}, Orig Reward: {accumulated_original_reward:.2f}, Steps: {episode_steps}, Duration: {episode_duration:.2f}s")
            wandb.log({
                "Episode/Episode Number": ep + 1,
                "Reward/Shaped Episode Reward": accumulated_shaped_reward,   # <--- Log shaped
                "Reward/Original Episode Reward": accumulated_original_reward, # <--- Log original
                "Episode/Steps": episode_steps,
                "Perf/Episode Duration (s)": episode_duration,
                "Progress/Env Steps": self.env_count,
                "Progress/Train Steps": self.train_count,
                "Parameters/Epsilon (End of Ep)": self.epsilon if not self.use_noisy else 0,
            }, step=self.env_count)

            # --- Periodic Evaluation ---
            eval_freq = getattr(self.args, "eval_frequency_episodes", 20)
            if (ep + 1) % eval_freq == 0  and self.env_count >= 50000:
                eval_reward = self.evaluate()
                wandb.log({"Reward/Evaluation Reward": eval_reward}, step=self.env_count)
                # --- Save Best Model ---
                # Refined saving condition
                if eval_reward >= self.best_reward:
                      self.best_reward = eval_reward
                      best_model_path = os.path.join(self.save_dir, "best_q_net.pt")
                      torch.save(self.q_net.state_dict(), best_model_path)
                      print(f"Saved new best model with eval reward {eval_reward:.2f} to {best_model_path}")


        # End of Training
        total_duration = time.time() - total_start_time
        print(f"\nTraining finished. Total duration: {total_duration:.2f} seconds")
        self.env.close()
        self.test_env.close()

    def train(self):
        """Perform a single training update step."""
        self.train_count += 1

        sample_result = self.memory.sample(self.batch_size)
        if sample_result is None: return
        states, actions, rewards, next_states, dones, weights, indices = sample_result

        states_t = torch.from_numpy(states).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        actions_t = actions.to(self.device)
        rewards_t = rewards.to(self.device)
        dones_t = dones.to(self.device)
        weights_t = weights.to(self.device)

        # --- Calculate Loss based on flags ---
        gamma_n_step = self.gamma ** self.n_step
        if self.use_distributional:
            elementwise_loss = self._compute_c51_loss( # Renamed for clarity
                states_t, actions_t, rewards_t, next_states_t, dones_t, gamma_n_step
            )
            loss = (elementwise_loss * weights_t).mean()
            td_errors_for_priority = elementwise_loss # Use C51 loss for priority
        else:
            # Calculate DDQN loss (e.g., MSE)
            loss, td_errors = self._compute_ddqn_loss( # New method needed
                 states_t, actions_t, rewards_t, next_states_t, dones_t, gamma_n_step, weights_t
            )
            td_errors_for_priority = td_errors # Use TD error for priority

        # --- Gradient descent ---
        self.optimizer.zero_grad()
        loss.backward()
        clip_value = getattr(self.args, "gradient_clip_value", 10.0)
        clip_grad_norm_(self.q_net.parameters(), clip_value)
        self.optimizer.step()

        # --- Update PER priorities ---
        self.memory.update_priorities(indices, td_errors_for_priority)

        # --- Reset noise if using Noisy Nets ---
        if self.use_noisy:
            self.q_net.reset_noise()
            self.target_net.reset_noise()

        # --- Update target network ---
        if self.train_count % self.target_update_frequency == 0:
            self._update_target_net()

        # --- Log training stats ---
        log_freq = getattr(self.args, "train_log_frequency", 1000)
        if self.train_count % log_freq == 0:
             # Calculate Q-value stats for logging
             with torch.no_grad():
                  q_values = self.q_net(states_t) # Get Q-values
                  q_mean = q_values.mean().item()
                  q_std = q_values.std().item()

             print(f"[Train] Step: {self.train_count}, Loss: {loss.item():.4f}, Q Mean: {q_mean:.3f}")
             wandb.log({
                 "Loss/Train Loss": loss.item(),
                 "Stats/Q-Value Mean (Train Batch)": q_mean,
                 "Stats/Q-Value Std (Train Batch)": q_std,
                 "Progress/Train Steps": self.train_count # Log train steps here too
             }, step=self.env_count)

    # Renamed loss function for clarity
    def _compute_c51_loss(self, states, actions, rewards, next_states, dones, gamma_n_step) -> torch.Tensor:
         """Return element-wise C51 loss (KL divergence)."""
         # Calculate target distribution projection
         with torch.no_grad():
             next_q_values = self.q_net(next_states)
             next_actions = next_q_values.argmax(1)
             next_dist_target = self.target_net.dist(next_states) # Network needs dist method
             next_dist = next_dist_target[range(self.batch_size), next_actions]

             t_z = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma_n_step * self.target_net.support
             t_z = t_z.clamp(min=self.v_min, max=self.v_max)
             delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
             b = (t_z - self.v_min) / delta_z
             l = b.floor().long()
             u = b.ceil().long()
             l = l.clamp(0, self.atom_size - 1)
             u = u.clamp(0, self.atom_size - 1)

             weight_l = next_dist * (u.float() - b)
             weight_u = next_dist * (b - l.float())

             proj_dist = torch.zeros_like(next_dist)
             offset = torch.arange(self.batch_size, device=self.device).unsqueeze(1) * self.atom_size
             offset = offset.expand(self.batch_size, self.atom_size)
             proj_dist.view(-1).index_add_(0, (l + offset).view(-1), weight_l.view(-1))
             proj_dist.view(-1).index_add_(0, (u + offset).view(-1), weight_u.view(-1))

         # Calculate current distribution
         dist_current = self.q_net.dist(states) # Network needs dist method
         log_p = torch.log(dist_current[range(self.batch_size), actions] + 1e-6) # Adjusted epsilon

         # Calculate KL divergence loss
         elementwise_loss = -(proj_dist * log_p).sum(1)
         return elementwise_loss

    # --- NEW METHOD for DDQN + N-step + PER Loss ---
    def _compute_ddqn_loss(self, states, actions, rewards, next_states, dones, gamma_n_step, weights):
        """Computes the N-step Double DQN loss with PER weights.
           Returns the final weighted loss tensor and element-wise TD errors.
        """
        # Get current Q values Q(s, a) from online network
        # Network forward pass handles normalization
        q_values = self.q_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1) # Shape: (N,)

        # Get target Q values using Double DQN logic
        with torch.no_grad():
             # Select actions using online network: a' = argmax_a Q(s', a)
             next_actions = self.q_net(next_states).argmax(1) # Shape: (N,)
             # Get Q values from target network for selected actions: Q_target(s', a')
             next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(-1)).squeeze(-1) # Shape: (N,)
             # Calculate target: R_n + gamma^n * Q_target(s', a') * (1 - D_n)
             target_q_values = rewards + (1 - dones) * gamma_n_step * next_q_values # Shape: (N,)

        # Calculate element-wise TD errors
        td_errors = target_q_values - q_values # Shape: (N,)

        # Calculate element-wise loss (e.g., MSE or Huber)
        # Using MSE similar to the original sample DQN
        elementwise_loss = td_errors ** 2 # Shape: (N,)

        # Calculate the final loss, weighted by PER weights
        loss = (elementwise_loss * weights).mean() # Scalar loss

        # Return scalar loss and TD errors (for priority update)
        return loss, td_errors.detach() # Detach TD errors as they are only for priority

    # evaluate method remains largely the same, ensure select_action handles eval mode correctly
    def evaluate(self, num_eval_episodes=30):
        # ... (keep implementation, self.q_net.eval() is set, select_action handles greedy) ...
         """Evaluate agent performance by running episodes in the test environment."""
         print(f"\nStarting evaluation for {num_eval_episodes} episodes...")
         eval_start_time = time.time()
         episode_rewards = []

         self.q_net.eval() # Set network to evaluation mode

         for i in range(num_eval_episodes):
             obs, _ = self.test_env.reset(seed=self.seed + 100 + i) # Use different seeds for eval
             state = self.preprocessor.reset(obs)
             done = False
             episode_reward = 0
             original_episode_reward = 0
             episode_steps = 0

             while not done and episode_steps < self.max_episode_steps: # Add step limit for safety
                 # In eval mode, NoisyNet uses mean weights - deterministic action
                 # If not using Noisy, epsilon is effectively 0 during eval logic (not implemented here, but assumed)
                 with torch.no_grad():
                     # Use select_action, but force greedy if epsilon-greedy is active
                     # Simpler: just use greedy policy directly for evaluation
                     state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                     q_values = self.q_net(state_t)
                     action = q_values.argmax(1).item()

                 next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                 done = terminated or truncated
                 original_episode_reward += reward
                 state = self.preprocessor.step(next_obs)
                 episode_steps += 1


             episode_rewards.append(original_episode_reward) # Log original reward
             print(f" - Eval Episode {i+1}: Orig Reward {original_episode_reward:.2f}")

         avg_reward = numpy.mean(episode_rewards)
         eval_duration = time.time() - eval_start_time
         print(f"Evaluation finished. Average Original Reward: {avg_reward:.2f}, Duration: {eval_duration:.2f}s")

         self.q_net.train() # Set network back to training mode
         # Reset noise after evaluation if using Noisy Nets
         if self.use_noisy:
             self.q_net.reset_noise()
             self.target_net.reset_noise()

         return avg_reward # Return average original reward

    # _update_target_net remains the same
    def _update_target_net(self):
         # ... (keep implementation) ...
         self.target_net.load_state_dict(self.q_net.state_dict())
# Main execution block
# Main execution block
# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rainbow DQN Agent Training for Fast Pong Convergence")

    # --- Environment Arguments ---
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5", help="Gym environment ID")
    parser.add_argument("--seed", type=int, default=777, help="Random seed")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames to stack")
    parser.add_argument("--max-episode-steps", type=int, default=10000, help="Maximum steps per episode")

    # --- Aggressive Training Arguments ---
    parser.add_argument("--episodes", type=int, default=1000, help="Total training episodes (adjust to ensure ~200k+ steps)") # Might need more episodes now
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (Increased)")
    parser.add_argument("--discount-factor", type=float, default=0.97, help="Discount factor (gamma) (Lowered)")
    parser.add_argument("--target-update-frequency", type=int, default=4000, help="Target network update frequency (train steps) (Lowered)")
    parser.add_argument("--gradient-clip-value", type=float, default=10.0, help="Gradient clipping value")
    parser.add_argument("--train-log-frequency", type=int, default=1000, help="Log training stats frequency (train steps)")

    # --- Replay Buffer Arguments ---
    parser.add_argument("--memory-size", type=int, default=50000, help="Replay buffer capacity")
    parser.add_argument("--replay-start-size", type=int, default=10000, help="Min env steps before training starts (Lowered)")

    # --- PER Arguments ---
    parser.add_argument("--alpha", type=float, default=0.5, help="PER alpha")
    parser.add_argument("--beta", type=float, default=0.4, help="PER initial beta")
    parser.add_argument("--reward-scale", type=float, default=1.0, help="PER reward scale for priority")

    # --- N-step Arguments ---
    parser.add_argument("--n-step", type=int, default=3, help="N-step return")

    # --- Reward Shaping Argument --- # <--- 新增
    parser.add_argument("--rally-reward", type=float, default=0.0005, help="Small positive reward for each step during a rally")

    # --- Distributional RL (C51) Arguments ---
    parser.add_argument("--v-min", type=float, default=-5.0, help="C51 v_min (for clipped/shaped rewards)")
    parser.add_argument("--v-max", type=float, default=5.0, help="C51 v_max (for clipped/shaped rewards)")
    parser.add_argument("--atom-size", type=int, default=51, help="C51 number of atoms")

    # --- Logging and Saving Arguments ---
    parser.add_argument("--save-dir", type=str, default="./rainbow_fast_results", help="Directory for results")
    parser.add_argument("--wandb-project-name", type=str, default="DLP-Lab5-Rainbow-Fast", help="Wandb project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--eval-frequency-episodes", type=int, default=5, help="Evaluation frequency (episodes)")
    parser.add_argument("--save-interval", type=int, default=100000, help="Snapshot saving frequency (env steps)")
    # Anneal beta/epsilon over ~180k training steps (target 200k env - 20k start)
    parser.add_argument("--total-train-steps", type=int, default=1000000, help="Estimated total training steps for annealing")
    parser.add_argument("--train-per-step", type=int, default=1)

    # --- Rainbow Component Switches ---
    parser.add_argument("--use-dueling", action='store_true', help="Enable Dueling")
    parser.add_argument("--use-noisy", action='store_true', help="Enable Noisy Nets")
    parser.add_argument("--use-distributional", action='store_true', help="Enable Distributional RL (C51)")

    # --- Epsilon-Greedy Parameters ---
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    # Decay over ~180k training steps, equivalent to env steps after warmup
    parser.add_argument("--epsilon-decay-steps", type=int, default=60000, help="Env steps to decay epsilon over")
    parser.add_argument("--epsilon-final", type=float, default=0.01)

    # Parse arguments and calculate derived epsilon rate
    args = parser.parse_args()
    # ... (epsilon decay calculation remains the same) ...
    if not args.use_noisy:
        if args.epsilon_decay_steps > 0:
             epsilon_decay_rate = (args.epsilon_start - args.epsilon_final) / args.epsilon_decay_steps
        else:
             epsilon_decay_rate = 0
        args.epsilon_decay_rate = epsilon_decay_rate
        print(f"Epsilon-greedy enabled: start={args.epsilon_start}, final={args.epsilon_final}, decay_steps={args.epsilon_decay_steps}, decay_rate={args.epsilon_decay_rate:.8f}")
    else:
         args.epsilon_decay_rate = 0
         print("Noisy Nets enabled for exploration.")


    # ... (rest of __main__: seed, wandb init, agent creation, run) ...
    seed_torch(args.seed)
    if args.wandb_run_name is None:
         timestamp = time.strftime('%Y%m%d-%H%M%S')
         flags = f"{'D' if args.use_dueling else ''}{'N' if args.use_noisy else 'e'}{'C' if args.use_distributional else 'Q'}"
         # Add shaping info to name
         args.wandb_run_name = f"{flags}-PER-N{args.n_step}-Shaped{args.rally_reward}_{args.env_name.split('/')[-1]}_{timestamp}"
    # ... wandb init ...
    try:
        wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=vars(args), save_code=True, reinit=True)
        print(f"Wandb initialized for run: {args.wandb_run_name}")
    except Exception as e:
        print(f"Wandb initialization failed: {e}. Training without wandb logging.")

    agent = DQNAgent(env_name=args.env_name, args=args)
    print(f"Starting training for {args.episodes} episodes...")
    agent.run(args.episodes)
    if wandb.run is not None: wandb.finish()
    print("Script finished.")
    
    