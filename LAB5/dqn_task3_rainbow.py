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
import gc
gym.register_envs(ale_py) # Keep environment registration
import pickle
# --- Limit Threading ---
NUM_THREADS = "4" # Or "1", "4" depending on your preference/system
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS # For MKL if used by numpy/scipy
torch.set_num_threads(int(NUM_THREADS))
torch.set_num_interop_threads(int(NUM_THREADS)) # Often 1 or 2 is sufficient
cv2.setNumThreads(0) # Disable OpenCV threading (might rely on underlying BLAS)
# --------------------

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
# SumTree (Refactored for Priorities Only)
class SumTree:
    """SumTree data structure for managing priorities."""
    write = 0 # Tracks the next position in the tree structure (leaf node)

    def __init__(self, capacity):
        self.capacity = capacity
        # Tree structure: Stores priorities. Size is 2*capacity - 1.
        self.tree = numpy.zeros(2 * capacity - 1)
        # No self.data here anymore
        self.n_entries = 0 # Number of entries currently in the structure

    def _propagate(self, idx, change):
        """Propagate priority changes up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find tree leaf index based on cumulative priority 's'."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree): # Reached a leaf node
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            # Ensure subtraction doesn't lead to negative s due to floating point issues
            # Though theoretically s should always be >= self.tree[left] here
            s_new = s - self.tree[left]
            return self._retrieve(right, s_new)

    def total(self):
        """Get the total priority (root node value)."""
        return self.tree[0]

    def add(self, p):
        """Store priority p in the tree structure.

        Returns:
            int: The data index (0 to capacity-1) corresponding to this priority entry.
        """
        idx = self.write + self.capacity - 1 # Tree index for the leaf
        data_idx = self.write              # Corresponding index for external data buffer

        self.update(idx, p) # Update the tree with the new priority

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0 # Wrap around if capacity is reached

        if self.n_entries < self.capacity:
            self.n_entries += 1

        return data_idx # Return the index for the data buffer

    def update(self, idx, p):
        """Update priority of a leaf node and propagate the change."""
        # Ensure idx is a valid leaf index
        if not (self.capacity - 1 <= idx < 2 * self.capacity - 1):
             # This case might happen if indices from sampling are passed incorrectly
             # Or if external logic provides a non-leaf index.
             # For internal use in add, it should be correct.
             # If called from update_priorities, the idx comes from sampling.
             print(f"Warning: Attempting to update non-leaf index {idx} in SumTree. Max leaf index: {2*self.capacity-2}")
             # Decide handling: raise error, return, or proceed cautiously?
             # Let's proceed but log warning. The caller (update_priorities) has a check too.

        # Ensure priority is positive
        p = max(p, 1e-6) # Avoid zero priority causing issues in propagation or sampling

        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """Get leaf index, priority value, and corresponding data index for a sample value 's'."""
        idx = self._retrieve(0, s) # Tree index of the leaf
        dataIdx = idx - self.capacity + 1 # Corresponding index in the data buffer (0 to capacity-1)
        # Basic check: dataIdx should be within [0, capacity-1]
        if not (0 <= dataIdx < self.capacity):
             # This indicates an issue in _retrieve or the sampling value 's'
             # Possibly s > total() or floating point issues in _retrieve
             print(f"Error: SumTree.get retrieved invalid data index {dataIdx} for s={s}, total={self.total()}, tree_idx={idx}. Clamping to valid range.")
             # Clamp index to valid range as a fallback, but investigate the root cause
             dataIdx = max(0, min(dataIdx, self.capacity - 1))

        return (idx, self.tree[idx], dataIdx) # Return tree_idx, priority, data_idx
    
# PrioritizedReplayBuffer (Refactored with Typed Arrays)
class PrioritizedReplayBuffer:
    def __init__(self, capacity, frame_stack, alpha=0.6, beta_start=0.4, total_train_steps=1000000, reward_scale=1.0, epsilon=1e-6):
        print(f"Initializing PrioritizedReplayBuffer with capacity={capacity}, frame_stack={frame_stack}")
        if reward_scale <= 0:
            print(f"Warning: reward_scale must be positive, received {reward_scale}. Setting to 1.0.")
            reward_scale = 1.0

        self.capacity = capacity
        self.frame_stack = frame_stack
        self.alpha = alpha
        self.beta = beta_start
        self.beta_final = 1.0
        self.epsilon = epsilon # Small value added to priorities
        self.reward_scale = reward_scale # For scaling errors before priority calculation

        # Data storage using typed NumPy arrays
        self.states = numpy.empty((capacity, frame_stack, 84, 84), dtype=numpy.uint8)
        self.next_states = numpy.empty((capacity, frame_stack, 84, 84), dtype=numpy.uint8)
        # Ensure action dtype matches later conversion (torch.int64)
        self.actions = numpy.empty(capacity, dtype=numpy.int64)
        self.rewards = numpy.empty(capacity, dtype=numpy.float32)
        # Store dones as boolean, convert to float later if needed
        self.dones = numpy.empty(capacity, dtype=numpy.bool_)

        # Priority management using SumTree
        self.tree = SumTree(capacity)

        # Beta annealing parameters
        if total_train_steps > 0:
            self.beta_increment_per_sampling = (self.beta_final - self.beta) / total_train_steps
        else:
            self.beta_increment_per_sampling = 0

        print(f"PER Beta annealing: start={self.beta}, final={self.beta_final}, increment={self.beta_increment_per_sampling:.8e} per step over {total_train_steps} steps")

    def __len__(self):
        """Return the current number of transitions in the buffer."""
        # Use SumTree's n_entries as it tracks filled slots
        return self.tree.n_entries

    def add(self, transition, error):
        """Add a new n-step transition and its initial TD error."""
        state, action, reward, next_state, done = transition

        # Calculate priority (ensure error is non-negative for priority calc)
        scaled_error = abs(error / self.reward_scale)
        priority = (scaled_error + self.epsilon) ** self.alpha
        priority = max(priority, self.epsilon) # Safeguard against zero

        # Add priority to SumTree and get the index for data storage
        data_idx = self.tree.add(priority)

        # Store transition components in typed arrays at the obtained index
        try:
            self.states[data_idx] = state
            self.actions[data_idx] = action
            self.rewards[data_idx] = reward
            self.next_states[data_idx] = next_state
            self.dones[data_idx] = done
        except IndexError:
             print(f"Error: Index {data_idx} out of bounds for buffer capacity {self.capacity} during add.")
             # This shouldn't happen if SumTree.add returns correct index
             # Could occur if capacity mismatch or logic error in SumTree.write
             # Consider raising an error or handling more gracefully.


    def sample(self, batch_size):
        """Sample a batch of transitions with priorities and IS weights."""
        # Anneal beta
        if self.beta < self.beta_final:
            self.beta = min(self.beta_final, self.beta + self.beta_increment_per_sampling)

        if self.tree.n_entries == 0:
            print("Error: Attempting to sample from an empty buffer.")
            return None

        # Use tree.n_entries for IS weight calculation
        current_size = self.tree.n_entries

        batch_indices = numpy.empty(batch_size, dtype=numpy.int32) # For data retrieval
        tree_indices = numpy.empty(batch_size, dtype=numpy.int32)  # For priority update
        priorities = numpy.empty(batch_size, dtype=numpy.float32)

        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            # Ensure s does not exceed total priority
            s = random.uniform(a, min(b, self.tree.total()))
            s = min(s, self.tree.total()) # Redundant clamp?

            try:
                tree_idx, priority, data_idx = self.tree.get(s)

                # Check if the retrieved data index corresponds to a filled slot
                # data_idx should be < self.tree.n_entries if SumTree is managed correctly
                # Add a check here for robustness, although SumTree.get should ideally handle this.
                if data_idx >= current_size:
                     print(f"Warning: Sampled data index {data_idx} >= current buffer size {current_size}. Retrying sample {i}.")
                     # This indicates a potential issue in SumTree sampling or state.
                     # Need a retry mechanism for this sample. Let's implement a simple retry:
                     attempts = 0
                     max_attempts = 10 # Limit retries
                     while data_idx >= current_size and attempts < max_attempts:
                         s = random.uniform(0, self.tree.total()) # Sample from whole range on retry
                         tree_idx, priority, data_idx = self.tree.get(s)
                         attempts += 1
                     if data_idx >= current_size:
                         print(f"Error: Failed to sample valid index after {max_attempts} retries. Aborting sample.")
                         return None # Abort sampling if retry fails

                batch_indices[i] = data_idx
                tree_indices[i] = tree_idx
                priorities[i] = priority

            except Exception as e:
                print(f"Error during SumTree.get in sampling loop: {e}. Aborting sample.")
                return None # Abort if error occurs

        # Calculate Importance Sampling weights
        sampling_probabilities = priorities / self.tree.total()
        # Use current_size (which is tree.n_entries) for N in the formula
        weights = numpy.power(current_size * sampling_probabilities, -self.beta)
        # Normalize weights
        if weights.max() > 1e-9:
            weights /= weights.max()
        else:
            weights = numpy.ones_like(weights) # Avoid division by zero if max is ~0

        # Retrieve data from typed arrays using batch_indices
        batch_states = self.states[batch_indices]
        batch_actions = self.actions[batch_indices]
        batch_rewards = self.rewards[batch_indices]
        batch_next_states = self.next_states[batch_indices]
        batch_dones = self.dones[batch_indices]

        # Convert relevant parts to Torch tensors
        # States remain numpy arrays, convert in agent.train
        actions_t = torch.from_numpy(batch_actions) # Already int64
        rewards_t = torch.from_numpy(batch_rewards) # Already float32
        # Convert dones (bool) to float32 tensor
        dones_t = torch.from_numpy(batch_dones).to(dtype=torch.float32)
        weights_t = torch.from_numpy(weights).to(dtype=torch.float32)

        # Return numpy states, torch tensors for others, and SumTree indices
        return batch_states, actions_t, rewards_t, batch_next_states, dones_t, weights_t, tree_indices


    def update_priorities(self, indices, errors):
        """Update priorities in the SumTree based on new TD errors."""
        if isinstance(errors, torch.Tensor):
            # Ensure errors are positive for priority calculation
            errors = errors.abs().detach().cpu().numpy()
        else:
            # Ensure numpy errors are positive
            errors = numpy.abs(numpy.array(errors))


        for i, idx in enumerate(indices): # idx is the SumTree index
            scaled_error = abs(errors[i] / self.reward_scale)
            priority = (scaled_error + self.epsilon) ** self.alpha
            # Ensure priority is not zero and use SumTree's internal update
            self.tree.update(idx, priority) # Pass SumTree index directly              
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
    env = gym.make(env_name, render_mode=render_mode, repeat_action_probability=0.25) # 
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
        self.env = make_env(env_name, seed=self.seed)
        self.test_env = make_env(env_name, seed=self.seed + 1)
        self.num_actions = self.env.action_space.n
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
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=args.adam_eps)

        # --- Setup Replay Buffer (PER enabled) ---
        self.memory = PrioritizedReplayBuffer(
            capacity=args.memory_size,
            frame_stack=args.frame_stack, # <<< Pass frame_stack
            alpha=args.alpha,
            beta_start=args.beta,
            total_train_steps=args.total_train_steps,
            reward_scale=args.reward_scale,
            epsilon=1e-6 # Default epsilon for PER buffer
        )
        print(f"Initialized PER buffer with capacity {args.memory_size}.")

        # ***** START EXPERT DATA LOADING *****
        if args.load_expert_data:
            print(f"\n--- Attempting to load expert data from: {args.load_expert_data} ---")
            try:
                # Load the pickled data
                with open(args.load_expert_data, 'rb') as f:
                    expert_experiences = pickle.load(f)

                if not isinstance(expert_experiences, list):
                     print("Warning: Loaded expert data is not a list. Skipping pre-loading.")
                     expert_experiences = []
                else:
                     print(f"Successfully loaded {len(expert_experiences)} expert transitions.")

            except FileNotFoundError:
                print(f"Error: Expert data file not found at {args.load_expert_data}. Skipping pre-loading.")
                expert_experiences = [] # Ensure it's an empty list if file not found
            except Exception as e:
                print(f"Error loading or unpickling expert data from {args.load_expert_data}: {e}. Skipping pre-loading.")
                expert_experiences = [] # Ensure it's an empty list on other errors

            # Add loaded experiences to the PER buffer
            if expert_experiences:
                print(f"Adding {len(expert_experiences)} expert transitions to PER...")
                # Use reward_scale as proxy for high error to get high initial priority
                # This corresponds to priority (abs(reward_scale/reward_scale)+eps)**alpha = (1+eps)**alpha
                initial_priority_proxy_error = self.memory.reward_scale
                added_count = 0
                skipped_count = 0

                for transition in expert_experiences[:int(args.replay_start_size/2)]:
                    # Basic check for transition format (optional but good practice)
                    if isinstance(transition, tuple) and len(transition) == 5:
                        # The transition format from generate_expert_data.py is
                        # (S, A, R_n, S_next_n, D_n)
                        # which matches what buffer.add expects as the 'transition' tuple.
                        self.memory.add(transition, initial_priority_proxy_error)
                        added_count += 1
                        if added_count % 10000 == 0:
                            print(f"  ... added {added_count}/{len(expert_experiences)} expert transitions")
                    else:
                        skipped_count += 1
                        if skipped_count == 1: # Only print warning once
                            print("Warning: Skipping invalid transition format in expert data.")

                if skipped_count > 0:
                    print(f"Warning: Skipped {skipped_count} invalid transitions.")
                print(f"Finished adding {added_count} expert transitions. PER buffer size now: {len(self.memory)} / {self.memory.capacity}")
                
                # Optional: Re-calculate PER beta increment based on potential total steps including pre-fill?
                # For simplicity, we usually keep the original total_train_steps for annealing.
            else:
                 print("No expert transitions loaded or added.")
            print("--- Finished expert data loading attempt ---")
        else:
            print("\nNo expert data file provided (--load-expert-data). Starting with an empty PER buffer.")
        # ***** END EXPERT DATA LOADING *****


        # --- N-step Buffer ---
        self.n_step = args.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)

        # Store other hyperparameters
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.target_update_frequency = args.target_update_frequency * args.train_per_step

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

            while not done and episode_steps < self.max_episode_steps:
                # --- Epsilon Decay ---
                if not self.use_noisy and self.env_count >= self.replay_start_size:
                    self.epsilon = max(self.epsilon_final, self.epsilon - self.epsilon_decay_rate)

                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                original_episode_reward += reward
                clipped_reward = numpy.clip(reward, -1, 1).item()
                next_state = self.preprocessor.step(next_obs)
                self.n_step_buffer.append((state, action, clipped_reward, next_state, done))

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

            # --- Log Episode Results ---
            print(f"[Episode End] Ep: {ep+1}/{episodes}, Orig Reward: {original_episode_reward:.2f}, Steps: {episode_steps}, Duration: {episode_duration:.2f}s")
            wandb.log({
                "Episode/Episode Number": ep + 1,
                "Reward/Original Episode Reward": original_episode_reward,
                "Episode/Steps": episode_steps,
                "Perf/Episode Duration (s)": episode_duration,
                "Progress/Env Steps": self.env_count, # Log again
                "Progress/Train Steps": self.train_count, # Log again
                "Parameters/Epsilon (End of Ep)": self.epsilon if not self.use_noisy else 0,
            }, step=self.env_count)

            # --- Periodic Evaluation ---
            eval_freq = getattr(self.args, "eval_frequency_episodes", 5)
            if (ep + 1) % eval_freq == 0:
                eval_reward = self.evaluate(num_eval_episodes=getattr(self.args, "eval_episodes", 30))
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
        if sample_result is None:
            # print("Skipping training step due to sampling failure.") # Optional log
            return

        # Unpack samples - states are numpy arrays, others are tensors
        states_np, actions_t, rewards_t, next_states_np, dones_t, weights_t, indices = sample_result

        # --- Convert numpy states to tensors and move all to device ---
        # Use torch.as_tensor for potential memory efficiency if arrays are already suitable
        # Convert uint8 states to float32 for network input (normalization happens in DQN forward)
        try:
             states_t = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
             next_states_t = torch.as_tensor(next_states_np, dtype=torch.float32, device=self.device)
             actions_t = actions_t.to(self.device)
             rewards_t = rewards_t.to(self.device)
             dones_t = dones_t.to(self.device)
             weights_t = weights_t.to(self.device)
        except Exception as e:
             print(f"Error during tensor conversion or moving to device: {e}")
             # Log details about shapes and types if error persists
             print(f"Shapes: states={states_np.shape}, next_states={next_states_np.shape}")
             print(f"Types: states={states_np.dtype}, next_states={next_states_np.dtype}")
             print(f"Actions: {actions_t.shape}, {actions_t.dtype}")
             # ... etc.
             return # Skip training step if conversion fails


        # --- Calculate Loss (rest of the logic remains the same) ---
        gamma_n_step = self.gamma ** self.n_step
        if self.use_distributional:
            # _compute_c51_loss expects tensors
            elementwise_loss = self._compute_c51_loss(
                states_t, actions_t, rewards_t, next_states_t, dones_t, gamma_n_step
            )
            loss = (elementwise_loss * weights_t).mean()
            td_errors_for_priority = elementwise_loss # Use C51 loss for priority
        else:
            # _compute_ddqn_loss expects tensors
            loss, td_errors = self._compute_ddqn_loss(
                states_t, actions_t, rewards_t, next_states_t, dones_t, gamma_n_step, weights_t
            )
            td_errors_for_priority = td_errors # Use TD error for priority

        # --- Gradient descent (remains the same) ---
        self.optimizer.zero_grad()
        loss.backward()
        clip_value = getattr(self.args, "gradient_clip_value", 10.0)
        clip_grad_norm_(self.q_net.parameters(), clip_value)
        self.optimizer.step()

        # --- Update PER priorities ---
        # update_priorities expects SumTree indices and errors (tensor or numpy)
        self.memory.update_priorities(indices, td_errors_for_priority)

        # --- Reset noise (remains the same) ---
        if self.use_noisy:
            self.q_net.reset_noise()
            self.target_net.reset_noise()

        # --- Update target network (remains the same) ---
        if self.train_count % self.target_update_frequency == 0:
            self._update_target_net()

        # --- Log training stats (remains the same, uses states_t for Q-value calc) ---
        log_freq = getattr(self.args, "train_log_frequency", 1000)
        if self.train_count % log_freq == 0:
            # Calculate Q-value stats for logging
            with torch.no_grad():
                q_values = self.q_net(states_t) # Get Q-values using the state tensor
                q_mean = q_values.mean().item()
                q_std = q_values.std().item()

            print(f"[Train] Step: {self.train_count}, Loss: {loss.item():.4f}, Q Mean: {q_mean:.3f}")
            wandb.log({
                "Loss/Train Loss": loss.item(),
                "Stats/Q-Value Mean (Train Batch)": q_mean,
                "Stats/Q-Value Std (Train Batch)": q_std,
                "Progress/Train Steps": self.train_count,
                "Stats/TD_error_mean": td_errors_for_priority.abs().mean().item(),
                "Stats/Buffer_Coverage": len(self.memory) / self.memory.capacity,
            }, step=self.env_count) # Log against env steps for alignment

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

        # --- Cleanup after evaluation ---
        try:
            self.test_env.close()
            print("Closed test_env.")
        except Exception as e:
            print(f"Warning: Error closing test_env: {e}")
        # Optional: Force garbage collection
        gc.collect()
        # Optional: Clear CUDA cache if using GPU
        torch.cuda.empty_cache()
        # --------------------------------

        self.q_net.train() # Set network back to training mode
        # Reset noise after evaluation if using Noisy Nets
        if self.use_noisy:
            self.q_net.reset_noise()
            self.target_net.reset_noise()

        return avg_reward

    # _update_target_net remains the same
    def _update_target_net(self):
         # ... (keep implementation) ...
         self.target_net.load_state_dict(self.q_net.state_dict())
# Main execution block

import argparse # Ensure argparse is imported if not already at the top

def str2bool(v):
    """Helper function to convert string representations of booleans to actual boolean values"""
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        # Raise an error if the input is not a recognizable boolean string
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rainbow DQN Agent Training")

    # --- Environment Arguments ---
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5", help="Gym environment ID (default: ALE/Pong-v5)")
    parser.add_argument("--seed", type=int, default=777, help="Random seed for reproducibility (default: 777)")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames to stack for state representation (default: 4)")
    parser.add_argument("--max-episode-steps", type=int, default=10000, help="Maximum steps per episode (default: 10000)")

    # --- Training Arguments ---
    parser.add_argument("--episodes", type=int, default=1000, help="Total number of training episodes (default: 1000)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for Adam optimizer (default: 6.25e-5)") # Common Rainbow LR
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--discount-factor", type=float, default=0.99, help="Discount factor (gamma) (default: 0.99)")
    parser.add_argument("--target-update-frequency", type=int, default=8000, help="Frequency (in train steps) to update target network (default: 8000)") # Common Rainbow setting
    parser.add_argument("--gradient-clip-value", type=float, default=10.0, help="Value for gradient clipping (default: 10.0)")
    parser.add_argument("--train-log-frequency", type=int, default=1000, help="Frequency (in train steps) to log training loss/stats (default: 1000)") # Added argument

    # --- Replay Buffer Arguments ---
    parser.add_argument("--memory-size", type=int, default=50000, help="Capacity of the replay buffer (default: 100k)") # Reduced from 1M for practicality
    parser.add_argument("--replay-start-size", type=int, default=50000, help="Minimum buffer size (env steps) before training starts (default: 20k)") # Reduced from 50k

    # --- Prioritized Experience Replay (PER) Arguments ---
    parser.add_argument("--alpha", type=float, default=0.5, help="PER alpha (prioritization exponent) (default: 0.5)") # Common Rainbow setting
    parser.add_argument("--beta", type=float, default=0.4, help="PER initial beta (importance sampling exponent) (default: 0.4)") # Anneals to 1.0
    parser.add_argument("--reward-scale", type=float, default=1.0, help="Reward scale factor used in PER priority calculation (default: 1.0)")

    # --- N-step Learning Arguments ---
    parser.add_argument("--n-step", type=int, default=3, help="N-step return calculation (default: 3)")

    # --- Distributional RL (C51) Arguments ---
    parser.add_argument("--v-min", type=float, default=-10.0, help="Minimum value of C51 support (default: -10 for general Atari)") # Adjusted default based on Rainbow paper
    parser.add_argument("--v-max", type=float, default=10.0, help="Maximum value of C51 support (default: 10 for general Atari)") # Adjusted default based on Rainbow paper
    parser.add_argument("--atom-size", type=int, default=51, help="Number of atoms in C51 support (default: 51)")

    # --- Logging and Saving Arguments ---
    parser.add_argument("--save-dir", type=str, default="./rainbow_results", help="Directory to save models and results (default: ./rainbow_results)")
    parser.add_argument("--wandb-project-name", type=str, default="DLP-Lab5-Rainbow", help="Wandb project name (default: DLP-Lab5-Rainbow)")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Wandb run name (defaults to agent type + timestamp)")
    parser.add_argument("--eval-frequency-episodes", type=int, default=5, help="Frequency (in episodes) to evaluate the agent (default: 20)")
    parser.add_argument("--save-interval", type=int, default=200000, help="Frequency (in env steps) to save model snapshots (default: 200k)")
    parser.add_argument("--total-train-steps", type=int, default=1000000, help="Estimated total training steps for beta annealing schedule (default: 1M)")

    # --- Rainbow Component Switches ---
    parser.add_argument("--use-dueling", type=str2bool, nargs='?', const=True, default=False,
                        help="Enable Dueling network architecture. Can be used like --use-dueling or --use-dueling=True/False. (default: False)")
    parser.add_argument("--use-noisy", type=str2bool, nargs='?', const=True, default=False,
                        help="Enable Noisy Nets (replaces epsilon-greedy). Can be used like --use-noisy or --use-noisy=True/False. (default: False)")
    parser.add_argument("--use-distributional", type=str2bool, nargs='?', const=True, default=False,
                        help="Enable Distributional RL (C51). Can be used like --use-distributional or --use-distributional=True/False. (default: False)")

    # --- Epsilon-Greedy Parameters (Used only if --use-noisy is False) ---
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon for epsilon-greedy (if not using Noisy Nets)")
    parser.add_argument("--epsilon-decay-steps", type=int, default=100000, help="Number of env steps to decay epsilon over (if not using Noisy Nets)")
    parser.add_argument("--epsilon-final", type=float, default=0.01, help="Final epsilon value (if not using Noisy Nets)")
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument(
        "--load-expert-data",
        type=str,
        default=None,
        help="Path to expert experience .pkl file to pre-load into PER. If provided, PER will be pre-filled."
    )
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes for each evaluation run (default: 10)")
    parser.add_argument("--adam-eps", type=float, default=1.5e-4, help="Adam epsilon for stability (default: 1.5e-4, similar to Dopamine)")

    # Parse the arguments ONCE
    args = parser.parse_args()

    # --- Derived Epsilon Decay Rate --- # Calculate and add AFTER parsing
    if not args.use_noisy:
        if args.epsilon_decay_steps > 0:
            epsilon_decay_rate = (args.epsilon_start - args.epsilon_final) / args.epsilon_decay_steps
        else:
            epsilon_decay_rate = 0
        args.epsilon_decay_rate = epsilon_decay_rate
        print(f"Epsilon-greedy enabled: start={args.epsilon_start}, final={args.epsilon_final}, decay_steps={args.epsilon_decay_steps}, decay_rate={args.epsilon_decay_rate:.8f}")
    else:
        args.epsilon_decay_rate = 0 # Assign 0 even if noisy is used, DQNAgent init expects it
        print("Noisy Nets enabled for exploration.")

    # Set random seeds
    seed_torch(args.seed)

    # Initialize Wandb
    # ... (wandb init code remains the same) ...
    if args.wandb_run_name is None:
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        flags = f"{'D' if args.use_dueling else ''}{'N' if args.use_noisy else 'e'}{'C' if args.use_distributional else 'Q'}" # e.g., DeC for Dueling+Eps+C51
        args.wandb_run_name = f"{flags}-N{args.n_step}_{args.lr}_{args.train_per_step}_{args.batch_size}_{args.discount_factor}_{args.target_update_frequency}_{args.replay_start_size}_{timestamp}"

    try:
        wandb.init(
            project=args.wandb_project_name,
            name=args.wandb_run_name,
            config=vars(args),
            save_code=True,
            reinit=True
        )
        print(f"Wandb initialized for run: {args.wandb_run_name}")
    except Exception as e:
        print(f"Wandb initialization failed: {e}. Training without wandb logging.")

    # Create the Rainbow DQN Agent
    agent = DQNAgent(env_name=args.env_name, args=args)

    # Start the training process
    print(f"Starting training for {args.episodes} episodes...")
    agent.run(args.episodes)

    print("Training finished. Cleaning up...")
    try:
        agent.env.close()
        print("Closed main env.")
        # test_env might have been closed in the last evaluate, but close again just in case
        if hasattr(agent, 'test_env'):
            agent.test_env.close()
            print("Closed test_env again.")
    except Exception as e:
        print(f"Warning: Error closing environments during final cleanup: {e}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache finally.")

    gc.collect()

    if wandb.run is not None:
        wandb.finish()
    print("Script finished.")