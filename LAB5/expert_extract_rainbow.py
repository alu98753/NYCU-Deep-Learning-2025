#!/usr/bin/env python3
# generate_expert_data.py
# Description: Loads a pre-trained Rainbow DQN model ('expert') and runs it
#              in the environment to collect and save ***1-step*** transitions
#              (s, a, r, s', d) for pre-filling a new agent's PER.
#              MODIFIED TO SAVE 1-STEP DATA.

import os
import random
import argparse
import time
import pickle             # To save the collected data
from collections import deque

import numpy              # Use 'numpy' to match training script style
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import cv2
import ale_py             # Required for gym.register_envs

# --- Reuse components from your training script ---
# (Copy these class/function definitions directly from your rainbow.py)

# --- Limit Threading ---
NUM_THREADS = "1" # Usually fine for data generation
os.environ["OMP_NUM_THREADS"] = NUM_THREADS
os.environ["MKL_NUM_THREADS"] = NUM_THREADS
torch.set_num_threads(int(NUM_THREADS))
torch.set_num_interop_threads(int(NUM_THREADS))
cv2.setNumThreads(0)
# --------------------

# --- NoisyLinear ---
# IMPORTANT: Make sure this definition MATCHES EXACTLY the one used
#            in your training script, especially the forward method!
class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet."""
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5): # Default 0.5 is common
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init # This will be overridden if loading state_dict

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters() # Initialize parameters
        # Note: reset_noise() samples initial noise, but forward pass controls usage

    def reset_parameters(self):
        mu_range = 1 / numpy.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        # Initialize sigma based on std_init - state_dict will overwrite this later
        self.weight_sigma.data.fill_(self.std_init / numpy.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / numpy.sqrt(self.out_features))

    def reset_noise(self):
        """Sample new noise buffers."""
        device = self.weight_mu.device
        epsilon_in = self._scale_noise(self.in_features).to(device)
        epsilon_out = self._scale_noise(self.out_features).to(device)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features).to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass. Uses noise only if in training mode. """
        if self.training:
            # Sample new noise before forward pass during training
            # self.reset_noise() # Typically called externally before batch update
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Use the mean weights/biases for deterministic action selection during eval
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        # Corrected: Generate tensor directly on CPU, move to device in reset_noise
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

# Helper function for weight initialization (copy from rainbow.py)
def init_weights(m):
    """Initialize weights using Kaiming uniform for Conv/Linear layers, but skip NoisyLinear."""
    # Skip NoisyLinear initialization here, it handles its own reset_parameters
    if isinstance(m, nn.Conv2d) or (isinstance(m, nn.Linear) and not isinstance(m, NoisyLinear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Atari Preprocessor (copy from rainbow.py)
class AtariPreprocessor:
    """Preprocessing the state input for Atari environments."""
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        """Convert to grayscale and resize."""
        if len(obs.shape) == 1:
            return obs
        else:
            # Ensure input is uint8 for cvtColor
            if obs.dtype != numpy.uint8:
                # Attempt conversion, warn if strange dtype
                try:
                    obs = obs.astype(numpy.uint8)
                except ValueError:
                    print(f"Warning: Unexpected observation dtype {obs.dtype}, attempting conversion to uint8.")
                    # Handle potential issues, e.g., scale if float
                    if np.issubdtype(obs.dtype, np.floating):
                         obs = (obs * 255).clip(0, 255).astype(numpy.uint8)
                    else: # Fallback if conversion is difficult
                         obs = obs.astype(numpy.uint8, errors='ignore')

            # Check if already grayscale
            if len(obs.shape) == 2:
                 gray = obs
            elif obs.shape[2] == 1:
                 gray = obs.squeeze(axis=2)
            elif obs.shape[2] == 3:
                 gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            else:
                 raise ValueError(f"Unexpected number of channels in observation: {obs.shape}")

            resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
            return resized.astype(numpy.uint8) # Ensure output is uint8

    def reset(self, obs):
        """Reset frame buffer with the first observation."""
        frame = self.preprocess(obs)
        if len(frame.shape) == 1:
             return frame # For non-image envs
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return numpy.stack(self.frames, axis=0)

    def step(self, obs):
        """Process a new observation and add to frame buffer."""
        frame = self.preprocess(obs)
        if len(frame.shape) == 1:
             return frame # For non-image envs
        self.frames.append(frame)
        return numpy.stack(self.frames, axis=0)

# DQN Network (copy from rainbow.py - ENSURE IT MATCHES your training script)
class DQN(nn.Module):
    """ Deep Q-Network with switchable components (Dueling, Noisy, C51). """
    def __init__(self, num_actions: int, frame_stack: int = 4,
                 use_dueling: bool = False, use_noisy: bool = False, use_distributional: bool = False,
                 atom_size: int = 51, v_min: float = -10.0, v_max: float = 10.0,
                 noisy_std_init: float = 0.5): # Add noisy_std_init
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.frame_stack = frame_stack
        self.use_dueling = use_dueling
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional
        self.atom_size = atom_size if use_distributional else 1
        self.v_min = v_min
        self.v_max = v_max
        # Use the provided noisy_std_init when creating NoisyLinear layers
        linear_layer = lambda in_f, out_f: NoisyLinear(in_f, out_f, std_init=noisy_std_init) if use_noisy else nn.Linear(in_f, out_f)

        if use_distributional:
            self.register_buffer("support", torch.linspace(self.v_min, self.v_max, self.atom_size))
        else:
            self.support = None # Set support to None if not distributional

        self.feature_layer = nn.Sequential(
            nn.Conv2d(self.frame_stack, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate feature_dim dynamically
        with torch.no_grad(): # No need for gradients here
             dummy_input = torch.zeros(1, frame_stack, 84, 84)
             feature_output = self.feature_layer(dummy_input)
             self.feature_dim = feature_output.shape[1]

        if self.use_dueling:
            self.advantage_hidden_layer = linear_layer(self.feature_dim, 512)
            adv_out_size = num_actions * self.atom_size
            self.advantage_layer = linear_layer(512, adv_out_size)

            self.value_hidden_layer = linear_layer(self.feature_dim, 512)
            val_out_size = self.atom_size
            self.value_layer = linear_layer(512, val_out_size)
        else:
            self.common_hidden_layer = linear_layer(self.feature_dim, 512)
            final_out_size = num_actions * self.atom_size
            self.final_layer = linear_layer(512, final_out_size)

        # Apply initialization to non-noisy linear layers if needed
        if not use_noisy:
             self.apply(init_weights) # Apply Kaiming init to Conv and standard Linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Calculates expected Q-values OR action distributions. """
        # Normalize input if it hasn't been done yet
        if x.dtype == torch.uint8:
             x = x.float() / 255.0
        elif x.max() > 1.0: # Basic check if likely not normalized
             x = x / 255.0

        feature = self.feature_layer(x)

        if self.use_dueling:
            adv_hid = F.relu(self.advantage_hidden_layer(feature))
            val_hid = F.relu(self.value_hidden_layer(feature))

            advantage = self.advantage_layer(adv_hid)
            value = self.value_layer(val_hid)

            # Reshape for atoms
            advantage = advantage.view(-1, self.num_actions, self.atom_size)
            value = value.view(-1, 1, self.atom_size)

            # Combine value and advantage streams
            q_logits = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            common_hid = F.relu(self.common_hidden_layer(feature))
            q_logits = self.final_layer(common_hid)
            q_logits = q_logits.view(-1, self.num_actions, self.atom_size)

        # --- Return expected Q-values for action selection ---
        if self.use_distributional:
            # Calculate probabilities (softmax over atoms)
            dist = F.softmax(q_logits, dim=-1) # Shape: (batch, num_actions, atom_size)
            # Calculate expected Q-value: sum(probability * atom_value)
            q_values = torch.sum(dist * self.support, dim=2) # Shape: (batch, num_actions)
        else:
            # If not distributional, the output is already Q-values (atom_size=1)
            q_values = q_logits.squeeze(-1) # Shape: (batch, num_actions)

        return q_values # Return expected Q-values

    def get_distribution(self, x: torch.Tensor) -> torch.Tensor:
         """ Calculates action value distribution (logits or probabilities). """
         if not self.use_distributional:
              raise RuntimeError("Distributional RL (C51) is not enabled for this model.")

         if x.dtype == torch.uint8:
              x = x.float() / 255.0
         elif x.max() > 1.0:
              x = x / 255.0

         feature = self.feature_layer(x)

         if self.use_dueling:
              adv_hid = F.relu(self.advantage_hidden_layer(feature))
              val_hid = F.relu(self.value_hidden_layer(feature))
              advantage = self.advantage_layer(adv_hid)
              value = self.value_layer(val_hid)
              advantage = advantage.view(-1, self.num_actions, self.atom_size)
              value = value.view(-1, 1, self.atom_size)
              q_logits = value + advantage - advantage.mean(dim=1, keepdim=True)
         else:
              common_hid = F.relu(self.common_hidden_layer(feature))
              q_logits = self.final_layer(common_hid)
              q_logits = q_logits.view(-1, self.num_actions, self.atom_size)

         # Return the logits or probabilities depending on what the loss function expects
         # For cross-entropy loss, return logits.
         return q_logits # Shape: (batch, num_actions, atom_size)


    def reset_noise(self): # Keep for consistency if Noisy Nets are used
        if self.use_noisy:
            for module in self.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()


# Seeding helper (copy from rainbow.py)
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    # Note: Environment seeding is separate

# Environment creation helper (copy from rainbow.py)
def make_env(env_name: str, seed: int = None, render_mode: str = "rgb_array"):
    """Creates the base Atari environment."""
    try:
        # Set repeat_action_probability to 0 for deterministic expert actions if desired
        env = gym.make(env_name, render_mode=render_mode, repeat_action_probability=0.0)
        print(f"Created environment '{env_name}' with render_mode='{render_mode}'.")
    except gym.error.NameNotFound:
        print(f"Environment {env_name} not found, attempting to register ALE environments.")
        gym.register_envs(ale_py)
        env = gym.make(env_name, render_mode=render_mode, repeat_action_probability=0.0)
        print(f"Created environment '{env_name}' after registration.")
    except Exception as e:
         print(f"Error creating environment '{env_name}': {e}")
         raise # Re-raise the exception

    # Seeding is handled by env.reset(seed=...)
    return env

# -------------------------------------------------------------
#  Main Data Generation Function (MODIFIED FOR 1-STEP)
# -------------------------------------------------------------
@torch.no_grad() # Disable gradients for inference
def generate_data(args):
    seed_torch(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Environment and Preprocessor ---
    env = make_env(args.env_name, seed=args.seed, render_mode="rgb_array")
    preprocessor = AtariPreprocessor(frame_stack=args.frame_stack)
    num_actions = env.action_space.n
    print(f"Environment: {args.env_name}, Actions: {num_actions}")

    # --- Build Network ---
    # Pass noisy_std_init from args to DQN constructor
    q_net = DQN(
        num_actions=num_actions, frame_stack=args.frame_stack,
        use_dueling=args.use_dueling, use_noisy=args.use_noisy, use_distributional=args.use_distributional,
        atom_size=args.atom_size, v_min=args.v_min, v_max=args.v_max,
        noisy_std_init=args.noisy_std_init # Pass the init value
    ).to(device)

    # --- Load Expert Model Weights ---
    print(f"Loading expert model weights from: {args.model_path}")
    try:
        # Load state dict - ensure strict=False if loading non-Rainbow into Rainbow structure or vice-versa,
        # but ideally the architectures should match exactly.
        q_net.load_state_dict(torch.load(args.model_path, map_location=device), strict=True)
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {args.model_path}")
        env.close()
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure the model architecture arguments match the saved checkpoint.")
        env.close()
        return

    q_net.eval() # Set to evaluation mode

    # --- Data Collection Loop ---
    expert_experiences = [] # List to store 1-step transitions

    # N-step buffer and gamma are NOT needed for saving 1-step data
    # n_step_buffer = deque(maxlen=args.n_step)
    # gamma = args.discount_factor

    total_steps_collected = 0 # Tracks number of 1-step transitions saved
    episode_count = 0

    start_time = time.time()
    print(f"Starting data generation for {args.num_steps} single-step transitions...")

    # Reset environment for the first time
    obs, _ = env.reset(seed=args.seed + episode_count) # Seed each episode differently
    state = preprocessor.reset(obs) # Get initial stacked state
    done = False

    while total_steps_collected < args.num_steps:
        # --- Select Action ---
        state_t = torch.from_numpy(state).to(device, dtype=torch.float32) # No unsqueeze needed if preprocessor outputs (C,H,W) or similar
        if len(state_t.shape) == 3: # Add batch dim if missing (C, H, W) -> (1, C, H, W)
             state_t = state_t.unsqueeze(0)
        elif len(state_t.shape) != 4: # Check for unexpected shape
             raise ValueError(f"Unexpected state shape for model input: {state_t.shape}")

        action = q_net(state_t).argmax(dim=1).item() # Get action index

        # --- Interact with Environment (using frame_skip) ---
        cumulative_reward = 0.0
        last_raw_obs = None
        terminated = False
        truncated = False
        info = {}
        for _ in range(args.frame_skip): # Use args.frame_skip
            next_obs_raw, reward, terminated, truncated, info = env.step(action)
            last_raw_obs = next_obs_raw
            cumulative_reward += reward
            current_done = terminated or truncated # Check done status after each frame skip step
            if current_done:
                break

        # --- Preprocess Next State ---
        next_state_processed = preprocessor.step(last_raw_obs)

        # --- Clip Reward (match training) ---
        clipped_reward = numpy.clip(cumulative_reward, -1.0, 1.0).item() # Clip the accumulated reward

        # --- **** Store 1-Step Transition **** ---
        # Store (s, a, r, s', d) where s and s' are processed states
        # Ensure data types match your replay buffer expectations
        expert_experiences.append((
            state.copy().astype(numpy.uint8),          # Current processed state s (uint8)
            numpy.uint8(action),                      # Action a (uint8 or int64)
            numpy.float32(clipped_reward),            # Clipped reward r (float32)
            next_state_processed.copy().astype(numpy.uint8), # Next processed state s' (uint8)
            numpy.float32(current_done)               # Done flag d (float32: 0.0 or 1.0)
        ))
        total_steps_collected += 1
        # --- **** Store End **** ---

        # --- Update State ---
        state = next_state_processed

        # --- Handle Episode End ---
        if current_done:
            episode_count += 1
            print(f"  Episode {episode_count} finished. Steps collected: {total_steps_collected}/{args.num_steps}")

            # Reset for next episode
            if total_steps_collected < args.num_steps:
                obs, _ = env.reset(seed=args.seed + episode_count)
                state = preprocessor.reset(obs)
                done = False
                # No need to clear n_step_buffer
            else:
                break # Stop if target steps reached

        # --- Progress Update ---
        if total_steps_collected % 5000 == 0 and total_steps_collected > 0:
            elapsed_time = time.time() - start_time
            print(f"Progress: {total_steps_collected}/{args.num_steps} single-step transitions saved. Time: {elapsed_time:.2f}s")


    # --- Save Collected Data ---
    print(f"\nFinished collecting {len(expert_experiences)} single-step transitions.")
    print(f"Saving expert experiences to: {args.output_file}")
    try:
        with open(args.output_file, 'wb') as f:
            pickle.dump(expert_experiences, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Experiences saved successfully.")
    except Exception as e:
        print(f"Error saving experiences: {e}")

    # --- Cleanup ---
    env.close()
    print("Environment closed.")

# -------------------------------------------------------------
#  Command Line Interface
# -------------------------------------------------------------
def str2bool(v):
    """Helper function to convert string representations of booleans to actual boolean values"""
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Expert 1-Step Trajectories using a Trained Rainbow DQN Model")

    # --- Required Arguments ---
    parser.add_argument("--model-path", required=True, type=str, help="Path to the trained expert model checkpoint (.pt)")
    parser.add_argument("--output-file", required=True, type=str, help="Path to save the collected 1-step expert experiences (.pkl)")

    # --- Environment and Data Collection Arguments ---
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5", help="Gym environment ID (must match expert's training)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment interaction")
    parser.add_argument("--num-steps", type=int, default=30000, # Default to 10k 1-step transitions
                        help="Total number of **single-step** transitions to collect")
    parser.add_argument("--frame-skip", type=int, default=4, help="Frame skip used during environment steps (match training)")


    # --- Core DQN/Rainbow Arguments (MUST match the loaded expert model's training config) ---
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames stacked (match expert)")
    # --discount-factor and --n-step are not directly used for saving 1-step data, but keep if model needs them
    # parser.add_argument("--discount-factor", type=float, default=0.99, help="Discount factor (gamma) (match expert)")
    # parser.add_argument("--n-step", type=int, default=3, help="N-step return config (match expert)")

    # --- Rainbow Component Switches (MUST match the loaded expert model's training config) ---
    parser.add_argument("--use-dueling", type=str2bool, nargs='?', const=True, default=True, help="Enable Dueling network architecture (match expert)") # Default True for Rainbow
    parser.add_argument("--use-noisy", type=str2bool, nargs='?', const=True, default=True, help="Enable Noisy Nets (match expert)") # Default True for Rainbow
    parser.add_argument("--noisy-std-init", type=float, default=0.5, help="Initial std dev for NoisyLinear (match expert if used)")
    parser.add_argument("--use-distributional", type=str2bool, nargs='?', const=True, default=True, help="Enable Distributional RL (C51) (match expert)") # Default True for Rainbow

    # --- Distributional RL (C51) Arguments (MUST match the loaded expert model if used) ---
    parser.add_argument("--v-min", type=float, default=-10.0, help="Minimum value of C51 support (match expert if distributional)")
    parser.add_argument("--v-max", type=float, default=10.0, help="Maximum value of C51 support (match expert if distributional)")
    parser.add_argument("--atom-size", type=int, default=51, help="Number of atoms in C51 support (match expert if distributional)")

    # This argument is for the training script, not needed here
    # parser.add_argument("--load-expert-data", type=str, default=None, help="Path to expert experience .pkl file to pre-load into PER.")

    args = parser.parse_args()

    # --- Run Data Generation ---
    generate_data(args)

    print("\nExpert 1-step data generation script finished.")