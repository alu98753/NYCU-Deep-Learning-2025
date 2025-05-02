#!/usr/bin/env python3
# -------------------------------------------------------------
#  Rainbow-DQN  evaluation script
#  - 支援 Dueling / NoisyLinear / C51  (與 train 時的 flags 對齊)
# -------------------------------------------------------------
import os, random, argparse, cv2, imageio
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py                                                     # noqa: F401  (僅為 gym 註冊)
from dqn_task3_duel_noise import AtariPreprocessor   ,NoisyLinear         # noqa: F401  (僅為 gym 註冊)


# ----------  Util  -------------------------------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ----------  Noisy Linear  ----------------------------------------

# ----------  Rainbow DQN (Inference)  --------------------------------
class DQN(nn.Module):
    """
    Dueling Deep Q Network with Noisy Linear Layers.
    """
    def __init__(self, num_actions, noisy_std=0.5): # Added noisy_std parameter
        super(DQN, self).__init__()
        self.num_actions = num_actions
        self.noisy_std = noisy_std

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

        # Calculate flattened feature size (manual calculation for 84x84 input)
        # Input (N, 4, 84, 84)
        # Conv1: (N, 32, 20, 20) # (84-8)/4 + 1 = 19+1=20
        # Conv2: (N, 64, 9, 9)   # (20-4)/2 + 1 = 8+1=9
        # Conv3: (N, 64, 7, 7)   # (9-3)/1 + 1 = 6+1=7
        # Flatten: (N, 64*7*7 = 3136)
        self.feature_size = 64 * 7 * 7

        # --- Dueling Streams using NoisyLinear ---
        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(self.feature_size, 512, std_init=self.noisy_std),
            nn.ReLU(),
            NoisyLinear(512, 1, std_init=self.noisy_std) # Output: V(s)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.feature_size, 512, std_init=self.noisy_std),
            nn.ReLU(),
            NoisyLinear(512, num_actions, std_init=self.noisy_std) # Output: A(s, a)
        )

    def forward(self, x):
        x = x / 255.0 # Normalize input images
        features = self.conv_layers(x)

        value = self.value_stream(features)          # V(s)
        advantages = self.advantage_stream(features) # A(s, a)

        # Combine value and advantages using Dueling formula:
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values

    def reset_noise(self):
        """Resets noise in all NoisyLinear layers."""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()



# ----------  評測主程式  ------------------------------------------
@torch.no_grad()
def evaluate(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(    args.env_name,    render_mode="rgb_array" )
    env.action_space.seed(args.seed)
    pre = AtariPreprocessor(args.frame_stack)
    n_actions = env.action_space.n
    net= DQN(n_actions, noisy_std=0.5).to(device)
    # ---- 建網路 & 載權重 ----
    # net = DQN(in_channels=args.frame_stack,
    #                  num_actions=n_actions,
    #                  frame_stack=args.frame_stack,
    #                  use_dueling=args.use_dueling,
    #                  use_noisy=args.use_noisy,
    #                  use_distributional=args.use_distributional,
    #                  atom_size=args.atom_size,
    #                  v_min=args.v_min, v_max=args.v_max).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()                        # eval() => NoisyLinear 取 μ (deterministic)
    # env = gym.make(    args.env_name,    render_mode="rgb_array",frameskip=4,    repeat_action_probability=0.002 )

    # if args.use_noisy:                # 若想要純 deterministic，可把 σ 清成 0
    #     for m in net.modules():
    #         if isinstance(m, NoisyLinear):
    #             m.weight_sig.fill_(0); m.bias_sig.fill_(0)

    os.makedirs(args.output_dir, exist_ok=True)
    cum_reward = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state  = pre.reset(obs)
        done   = False
        frames = []
        ep_reward = 0
        a = True
        while not done:
            frames.append(env.render())

            s_t = torch.from_numpy(state).float().unsqueeze(0).to(device)
            action = net(s_t).argmax(dim=1).item()
            if a:
                print(f"action: {action}")
                a = False

            next_obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_reward += r
            state = pre.step(next_obs)

        out = os.path.join(args.output_dir, f"ep{ep:03d}_{ep_reward:.0f}.mp4")
        with imageio.get_writer(out, fps=30) as vid:
            for f in frames: vid.append_data(f)
        print(f"[✓] Episode {ep} | R = {ep_reward:.0f} | saved → {out}")
        cum_reward += ep_reward

    print(f"\nAverage reward over {args.episodes} eps: {cum_reward/args.episodes:.2f}")
    env.close(); cv2.destroyAllWindows()

# ----------  CLI  --------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # 必填
    p.add_argument("--model-path", required=True,  help=".pt checkpoint from training")
    # 基本
    p.add_argument("--env-name",   default="ALE/Pong-v5")
    p.add_argument("--episodes",   type=int, default=20)
    p.add_argument("--seed",       type=int, default=777)
    p.add_argument("--output-dir", default="./eval_videos")
    p.add_argument("--frame-stack",type=int, default=4)
    # Rainbow flags – 必須和訓練時完全相同!
    p.add_argument("--use-dueling",        action="store_true")
    p.add_argument("--use-noisy",          action="store_true")
    p.add_argument("--use-distributional", action="store_true")
    p.add_argument("--atom-size", type=int, default=51)
    p.add_argument("--v-min",     type=float, default=-5.0)
    p.add_argument("--v-max",     type=float, default= 5.0)
    args = p.parse_args()

    evaluate(args)
