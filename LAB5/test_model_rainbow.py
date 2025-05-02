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
from dqn_task3_rainbow import AtariPreprocessor            # noqa: F401  (僅為 gym 註冊)


# ----------  Util  -------------------------------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# ----------  Noisy Linear  ----------------------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight_mu   = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sig  = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu     = nn.Parameter(torch.empty(out_features))
        self.bias_sig    = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))
        self.std_init = std_init
        self.reset_parameters(); self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sig.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sig.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps  .copy_(eps_out)

    @staticmethod
    def _scale_noise(size):                     # sign(randn) * sqrt(|randn|)
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:   # train => 隨機探索,  eval => 固定 mean
            w = self.weight_mu + self.weight_sig * self.weight_eps
            b = self.bias_mu   + self.bias_sig  * self.bias_eps
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

# ----------  Rainbow DQN (Inference)  --------------------------------
class RainbowNet(nn.Module):
    def __init__(self, in_channels, num_actions,
                 frame_stack=4,
                 use_dueling=False,
                 use_noisy=False,
                 use_distributional=False,
                 atom_size=51,
                 v_min=-10.0,
                 v_max=10.0):
        super().__init__()
        self.num_actions        = num_actions
        self.use_dueling        = use_dueling
        self.use_noisy          = use_noisy
        self.use_distributional = use_distributional
        self.atom_size          = atom_size if use_distributional else 1
        self.v_min, self.v_max  = v_min, v_max

        linear = NoisyLinear if use_noisy else nn.Linear
        if use_distributional:
            self.register_buffer("support",
                                 torch.linspace(v_min, v_max, self.atom_size))

        # === 注意名稱：feature_layer ===
        self.feature_layer = nn.Sequential(
            nn.Conv2d(frame_stack, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),         nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),         nn.ReLU(),
            nn.Flatten()
        )
        self.feature_dim = 64 * 7 * 7

        if use_dueling:
            # —— Advantage stream ——（名稱與訓練一致）
            self.advantage_hidden_layer = linear(self.feature_dim, 512)
            self.advantage_layer        = linear(512, num_actions * self.atom_size)
            # —— Value stream ——
            self.value_hidden_layer = linear(self.feature_dim, 512)
            self.value_layer        = linear(512,             self.atom_size)
        else:
            # === 注意名稱：common_hidden_layer, final_layer ===
            self.common_hidden_layer = linear(self.feature_dim, 512)
            self.final_layer         = linear(512, num_actions * self.atom_size)

    def forward(self, x: torch.Tensor):
        x = x / 255.0
        f = self.feature_layer(x)

        if self.use_dueling:
            adv_h = F.relu(self.advantage_hidden_layer(f))
            val_h = F.relu(self.value_hidden_layer(f))

            adv = self.advantage_layer(adv_h).view(-1, self.num_actions, self.atom_size)
            val = self.value_layer(val_h).view(-1, 1,             self.atom_size)
            logits = val + adv - adv.mean(1, keepdim=True)
        else:
            h = F.relu(self.common_hidden_layer(f))
            logits = self.final_layer(h).view(-1, self.num_actions, self.atom_size)

        if self.use_distributional:
            prob = F.softmax(logits, dim=-1).clamp(min=1e-3)
            q    = (prob * self.support).sum(dim=-1)       # expectation
            return q
        else:
            return logits.squeeze(-1)



# ----------  評測主程式  ------------------------------------------
@torch.no_grad()
def evaluate(args):
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(    args.env_name,    render_mode="rgb_array",frameskip=4,    repeat_action_probability=0.002 )
    env.action_space.seed(args.seed)
    pre = AtariPreprocessor(args.frame_stack)
    n_actions = env.action_space.n

    # ---- 建網路 & 載權重 ----
    net = RainbowNet(in_channels=args.frame_stack,
                     num_actions=n_actions,
                     frame_stack=args.frame_stack,
                     use_dueling=args.use_dueling,
                     use_noisy=args.use_noisy,
                     use_distributional=args.use_distributional,
                     atom_size=args.atom_size,
                     v_min=args.v_min, v_max=args.v_max).to(device)
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
