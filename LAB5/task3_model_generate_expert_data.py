# generate_expert_data.py
# Description: Runs a trained DuelingC51DQN agent and collects transitions
#              only from episodes achieving a minimum score threshold,
#              optionally filtering by maximum episode steps as well.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import gymnasium as gym
import cv2
# import ale_py
import os
from collections import deque
import argparse
import time

# --- AtariPreprocessor Class Definition --- (Unchanged)
class AtariPreprocessor:
    def __init__(self, frame_stack=4): self.frame_stack = frame_stack; self.frames = deque(maxlen=frame_stack)
    def preprocess(self, obs): # ... (code as before) ...
        if not isinstance(obs, np.ndarray): obs = np.array(obs)
        if len(obs.shape) == 3 and obs.shape[2] == 3: gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif len(obs.shape) == 2: gray = obs
        elif len(obs.shape) == 3 and obs.shape[2] == 1: gray = obs.squeeze(axis=2)
        else: raise ValueError(f"Unexpected obs shape: {obs.shape}")
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA); return resized.astype(np.uint8)
    def reset(self, obs): frame = self.preprocess(obs); self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack); return np.stack(self.frames, axis=0)
    def step(self, obs): frame = self.preprocess(obs); self.frames.append(frame); return np.stack(self.frames, axis=0)

# --- DuelingC51DQN Class Definition --- (Unchanged)
class DuelingC51DQN(nn.Module): # ... (code as before) ...
    def __init__(self, num_actions, num_atoms=51, vmin=-10, vmax=10): super(DuelingC51DQN, self).__init__(); self.num_actions = num_actions; self.num_atoms = num_atoms; self.vmin = vmin; self.vmax = vmax; self.support = torch.linspace(vmin, vmax, num_atoms); self.delta_z = (vmax - vmin) / (num_atoms - 1); self.conv_base = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(), nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(), nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(), nn.Flatten()); self.flattened_size = 64 * 7 * 7; self.value_stream = nn.Sequential(nn.Linear(self.flattened_size, 512), nn.ReLU(), nn.Linear(512, self.num_atoms)); self.advantage_stream = nn.Sequential(nn.Linear(self.flattened_size, 512), nn.ReLU(), nn.Linear(512, self.num_actions * self.num_atoms)); self.register_buffer("support_buf", self.support); self.register_buffer("delta_z_buf", torch.tensor(self.delta_z))
    def forward(self, x): x = x / 255.0; features = self.conv_base(x); value_logits = self.value_stream(features); advantage_logits = self.advantage_stream(features); value_logits = value_logits.view(-1, 1, self.num_atoms); advantage_logits = advantage_logits.view(-1, self.num_actions, self.num_atoms); mean_advantage_logits = advantage_logits.mean(1, keepdim=True); q_logits = value_logits + advantage_logits - mean_advantage_logits; return q_logits
    def get_expected_q_values(self, x): q_logits = self.forward(x); q_probs = F.softmax(q_logits, dim=2); expected_q = torch.sum(q_probs * self.support_buf, dim=2); return expected_q


def generate_expert_data(args):
    """Loads a model and generates expert transitions based on score threshold
       and optionally maximum episode steps."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup --- (Unchanged)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    try: env = gym.make(args.env_name, frameskip=1, render_mode=None)
    except Exception as e: print(f"Error creating env: {e}"); return
    env.action_space.seed(args.seed); preprocessor = AtariPreprocessor(frame_stack=args.frame_stack); num_actions = env.action_space.n

    # --- Load Model --- (Unchanged)
    if not os.path.exists(args.model_path): print(f"Error: Model path not found: {args.model_path}"); env.close(); return
    model = DuelingC51DQN(num_actions, args.num_atoms, args.vmin, args.vmax).to(device)
    try: model.load_state_dict(torch.load(args.model_path, map_location=device)); model.eval(); print(f"Successfully loaded model from {args.model_path}")
    except Exception as e: print(f"Error loading model state dict: {e}"); env.close(); return

    # --- Data Collection Loop --- (Modified Filtering Logic)
    collected_count = 0; total_episodes_run = 0
    # Slightly adjust heuristic limit maybe based on filtering? Hard to say, keep as is for now.
    max_episodes_to_try = args.num_transitions * 50 // (args.max_episode_steps or 1000) + 100

    all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []

    # <<< Determine if step filtering is active >>>
    use_step_filter = args.max_expert_episode_steps > 0
    filter_desc = f"score >= {args.min_score}"
    if use_step_filter:
        filter_desc += f" AND decision_steps <= {args.max_expert_episode_steps}"
    print(f"Starting data collection. Target: {args.num_transitions} transitions from episodes with {filter_desc}")

    while collected_count < args.num_transitions :
        total_episodes_run += 1
        obs, _ = env.reset(seed=args.seed + total_episodes_run)
        state = preprocessor.reset(obs)
        episode_states, episode_actions, episode_rewards, episode_next_states, episode_dones = [], [], [], [], []
        if "Pong" in args.env_name: fire_action = 1; fire_obs, _, _, _, _ = env.step(fire_action); state = preprocessor.step(fire_obs)
        done = False; episode_raw_reward = 0.0; decision_step_count = 0

        while not done:
            if args.max_episode_steps > 0 and decision_step_count >= args.max_episode_steps: break
            current_decision_step_state_np = np.array(state, dtype=np.uint8)
            # Select action (unchanged)
            state_tensor = torch.from_numpy(current_decision_step_state_np.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad(): expected_q = model.get_expected_q_values(state_tensor); action = expected_q.argmax().item()
            # Frame Skip Logic (unchanged)
            accumulated_raw_reward = 0.0; frame_done = False; last_obs = None
            for _ in range(args.frame_skip): #... (inner loop unchanged) ...
                next_obs, reward, terminated, truncated, _ = env.step(action); accumulated_raw_reward += reward; frame_done = terminated or truncated; last_obs = next_obs;
                if frame_done: break;
            if last_obs is None: print("Warning: last_obs is None..."); break;
            next_state = preprocessor.step(last_obs); next_state_np = np.array(next_state, dtype=np.uint8); done_flag_int = 1 if frame_done else 0
            # Store transition temporarily (unchanged)
            episode_states.append(current_decision_step_state_np); episode_actions.append(action); episode_rewards.append(accumulated_raw_reward); episode_next_states.append(next_state_np); episode_dones.append(done_flag_int)
            state = next_state; done = frame_done; episode_raw_reward += accumulated_raw_reward; decision_step_count += 1

        # --- <<< Episode End: Modified Filtering Logic >>> ---
        print(f"Episode {total_episodes_run} finished. Score: {episode_raw_reward:.0f}, Steps: {decision_step_count}. ", end="")

        # Check score condition first
        score_met = episode_raw_reward >= args.min_score
        steps_met = True # Assume steps condition met by default
        # If step filtering is active, check step condition
        if use_step_filter:
            steps_met = decision_step_count <= args.max_expert_episode_steps

        # Keep data only if ALL required conditions are met
        if score_met and steps_met:
            num_episode_transitions = len(episode_states)
            all_states.extend(episode_states); all_actions.extend(episode_actions); all_rewards.extend(episode_rewards); all_next_states.extend(episode_next_states); all_dones.extend(episode_dones)
            collected_count += num_episode_transitions
            print(f"KEPT (+{num_episode_transitions} transitions, Criteria Met). Total: {collected_count}/{args.num_transitions}")
        else:
            reason = ""
            if not score_met: reason += f"Score < {args.min_score}"
            if use_step_filter and not steps_met: reason += (" AND " if reason else "") + f"Steps > {args.max_expert_episode_steps}"
            print(f"DISCARDED ({reason}). Total: {collected_count}/{args.num_transitions}")
        # --- <<< End Modified Filtering Logic >>> ---

    env.close()

    # --- Save Collected Data --- (Unchanged)
    if collected_count >= args.num_transitions: # ... (Shuffle, trim, save logic) ...
        print(f"\nTarget number of transitions reached ({collected_count}). Saving data...")
        indices = list(range(collected_count)); random.shuffle(indices); indices = indices[:args.num_transitions]
        states_np=np.array([all_states[i] for i in indices],dtype=np.uint8); actions_np=np.array([all_actions[i] for i in indices],dtype=np.int64); rewards_np=np.array([all_rewards[i] for i in indices],dtype=np.float32); next_states_np=np.array([all_next_states[i] for i in indices],dtype=np.uint8); dones_np=np.array([all_dones[i] for i in indices],dtype=np.uint8)
        try: os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True); np.savez_compressed(args.output_file, states=states_np, actions=actions_np, rewards=rewards_np, next_states=next_states_np, dones=dones_np); print(f"Successfully saved {len(states_np)} filtered expert transitions to {args.output_file}");
        except Exception as e: print(f"Error saving data to {args.output_file}: {e}")
    elif total_episodes_run >= max_episodes_to_try: # ... (Partial save logic) ...
        print(f"\nWarning: Reached max episodes ({max_episodes_to_try})...")
        if collected_count>0 and input("Save partially collected data? (y/n): ").lower()=='y': 
            indices=list(range(collected_count)); random.shuffle(indices)
            states_np=np.array([all_states[i] for i in indices],dtype=np.uint8); actions_np=np.array([all_actions[i] for i in indices],dtype=np.int64); rewards_np=np.array([all_rewards[i] for i in indices],dtype=np.float32); next_states_np=np.array([all_next_states[i] for i in indices],dtype=np.uint8); dones_np=np.array([all_dones[i] for i in indices],dtype=np.uint8)
            try: os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True); np.savez_compressed(args.output_file, states=states_np, actions=actions_np, rewards=rewards_np, next_states=next_states_np, dones=dones_np); print(f"Saved {len(states_np)} transitions to {args.output_file}")
            except Exception as e: print(f"Error saving data: {e}")
    else: print(f"\nWarning: Data collection stopped unexpectedly. Collected {collected_count}/{args.num_transitions}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate expert demonstration data from a trained agent, filtering by episode score and optionally max steps.")

    # --- Required Arguments ---
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained DuelingC51DQN .pt model")
    parser.add_argument("--num-transitions", type=int, required=True, help="Target number of expert transitions to collect")

    # --- Environment Arguments ---
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5", help="Gym environment ID")
    parser.add_argument("--frame-stack", type=int, default=4, help="Number of frames stacked (must match model)")
    parser.add_argument("--frame-skip", type=int, default=4, help="Number of frames to skip per action decision")
    parser.add_argument("--max-episode-steps", type=int, default=27000 // 4, help="Max decision steps per episode during generation")

    # --- Model Architecture Arguments (MUST match loaded model) ---
    parser.add_argument("--num-atoms", type=int, default=51, help="Number of atoms used for the C51 model being loaded")
    parser.add_argument("--vmin", type=float, default=-5.0, help="Minimum value of the C51 support for the loaded model")
    parser.add_argument("--vmax", type=float, default=5.0, help="Maximum value of the C51 support for the loaded model")

    # --- Data Generation Control Arguments ---
    parser.add_argument("--output-file", type=str, default="expert_data_filteredB.npz", help="Filename for saving the generated expert data")
    parser.add_argument("--min-score", type=float, default=19.0, help="Minimum episode score required to keep the transitions") # Default 19
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation run")

    # <<< --- 新增參數：最大專家回合步數 --- >>>
    parser.add_argument("--max-expert-episode-steps", type=int, default=0,
                        help="Maximum decision steps allowed for an episode's transitions to be kept (if score is also met). "
                             "Set to 0 or negative to disable step filtering (only filter by min-score). Example: 2300")
    # <<< --------------------------------- >>>

    args = parser.parse_args()

    # --- Vmin/Vmax validation ---
    if args.vmin >= args.vmax:
        raise ValueError("--vmin must be strictly less than --vmax")

    generate_expert_data(args)