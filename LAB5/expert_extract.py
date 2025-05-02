import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import cv2
import os
import pickle
from collections import deque
import argparse


class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # normalize inside model
        return self.network(x / 255.0)


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
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


def generate_expert_data(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor(frame_stack=4)
    frame_skip = args.frame_skip

    num_actions = env.action_space.n
    model = DQN(4, num_actions).to(device)

    # load weights
    checkpoint = torch.load(args.model_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    model.eval()

    processed_data = []
    total_collected = 0
    ep_idx = 0
    episode_rewards = []

    print(f"Generating up to {args.num_transitions} transitions, frame_skip={frame_skip}")

    while total_collected < args.num_transitions:
        obs, _ = env.reset(seed=args.seed + ep_idx)
        state = preprocessor.reset(obs)
        done = False
        raw_transitions = []
        total_raw_reward = 0.0

        # collect raw transitions one per frame (inference unchanged)
        while not done:
            st = state.copy()
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            next_state = preprocessor.step(next_obs)

            raw_transitions.append(
                (st.astype(np.uint8), int(action), float(reward), next_state.astype(np.uint8), float(done))
            )
            total_raw_reward += reward
            state = next_state

        ep_idx += 1
        print(f"[Ep {ep_idx}] total raw reward = {total_raw_reward:.2f}")

        if total_raw_reward > 19:
            episode_rewards.append(total_raw_reward)
            # aggregate into frame_skip blocks + reward clipping
            for i in range(0, len(raw_transitions), frame_skip):
                if total_collected >= args.num_transitions:
                    break
                block = raw_transitions[i:i+frame_skip]
                if not block:
                    break
                s0, a0, *_, = block[0]
                ss, aa, rr, ns, dd = block[-1]
                sum_r = sum(r for (_, _, r, _, _) in block)
                # clip
                clipped_r = max(-1.0, min(sum_r, 1.0))
                done_block = any(d for (_, _, _, _, d) in block)
                processed_data.append((s0, a0, float(clipped_r), ns, float(done_block)))
                total_collected += 1
            print(f"  -> kept {min(len(raw_transitions)//frame_skip, args.num_transitions-total_collected)} aggregated transitions")
            print(f"  collected: {total_collected}/{args.num_transitions}")

    # save
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    print(f"Saved {len(processed_data)} transitions to {args.output_path}")

    if episode_rewards:
        print(f"Avg raw reward of episodes kept: {np.mean(episode_rewards):.2f}")

    env.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output-path', default='./expert_data_pong.pkl')
    parser.add_argument('--num-transitions', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--frame-skip', type=int, default=4,
                        help='Frames skipped per action when saving')
    args = parser.parse_args()
    generate_expert_data(args)
