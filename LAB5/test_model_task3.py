import torch
import torch.nn as nn
import torch.nn.functional as F # <<< 需要 F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse

# <<< 複製 DuelingC51DQN 類的定義過來 >>>
class DuelingC51DQN(nn.Module):
    def __init__(self, num_actions, num_atoms=51, vmin=-10, vmax=10):
        super(DuelingC51DQN, self).__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.vmin = vmin
        self.vmax = vmax

        # Calculate C51 support atoms
        self.support = torch.linspace(vmin, vmax, num_atoms)
        self.delta_z = (vmax - vmin) / (num_atoms - 1)

        # Shared convolutional base (same as original DQN)
        self.conv_base = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), # 輸入通道數固定為 4 (frame_stack)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.flattened_size = 64 * 7 * 7 # 3136

        # Dueling Streams
        self.value_stream = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_atoms)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions * self.num_atoms)
        )
        self.register_buffer("support_buf", self.support)
        self.register_buffer("delta_z_buf", torch.tensor(self.delta_z)) # delta_z 不需要 buffer

    def forward(self, x):
        x = x / 255.0
        features = self.conv_base(x)
        value_logits = self.value_stream(features)
        advantage_logits = self.advantage_stream(features)
        value_logits = value_logits.view(-1, 1, self.num_atoms)
        advantage_logits = advantage_logits.view(-1, self.num_actions, self.num_atoms)
        mean_advantage_logits = advantage_logits.mean(1, keepdim=True)
        q_logits = value_logits + advantage_logits - mean_advantage_logits
        return q_logits # 返回 logits

    def get_expected_q_values(self, x):
        """ Helper function to get expected Q-values for action selection """
        q_logits = self.forward(x)
        q_probs = F.softmax(q_logits, dim=2)
        expected_q = torch.sum(q_probs * self.support_buf, dim=2) # 使用 buffer
        return expected_q

# <<< 複製 AtariPreprocessor 類的定義過來 (確保是最新版本) >>>
class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if not isinstance(obs, np.ndarray):
             obs = np.array(obs)
        if len(obs.shape) == 3 and obs.shape[2] == 3: # Check for RGB
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        elif len(obs.shape) == 2: # Already grayscale
            gray = obs
        elif len(obs.shape) == 3 and obs.shape[2] == 1: # Grayscale with channel dim
             gray = obs.squeeze(axis=2)
        else:
             raise ValueError(f"Unexpected observation shape: {obs.shape}")

        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized.astype(np.uint8) # <<< 確保輸出 uint8

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0) # Shape: (4, 84, 84)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame) # append 新 frame
        return np.stack(self.frames, axis=0) # Shape: (4, 84, 84)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # <<< 使用 render_mode="rgb_array" 以便擷取畫面 >>>
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array", frameskip=1) # <<< 內部 frameskip 設為 1，我們手動控制 >>>
    env.action_space.seed(args.seed)
    # env.observation_space.seed(args.seed) # 可能不需要設定 observation space seed

    # <<< 建立 Preprocessor，使用固定的 frame_stack=4 >>>
    preprocessor = AtariPreprocessor(frame_stack=4)
    num_actions = env.action_space.n

    # <<< 實例化 DuelingC51DQN 模型 >>>
    model = DuelingC51DQN(
        num_actions,
        num_atoms=args.num_atoms,
        vmin=args.vmin,
        vmax=args.vmax
    ).to(device)

    # 載入模型權重
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model from {args.model_path}")
    except Exception as e:
        print(f"Error loading model from {args.model_path}: {e}")
        env.close()
        return

    model.eval() # 設定為評估模式

    os.makedirs(args.output_dir, exist_ok=True)
    all_rewards = 0
    frame_skip = args.frame_skip # <<< 從參數獲取 frame_skip 值

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = preprocessor.reset(obs) # (4, 84, 84) uint8
        done = False
        total_reward = 0
        frames = [] # 儲存影片的幀
        episode_steps = 0

        # <<< Pong 特有的 FIRE 動作 >>>
        if "Pong" in args.env_name:
             # 執行 FIRE 動作 (假設 action 1 是 FIRE) 並更新狀態
             fire_action = 1
             fire_obs, _, _, _, _ = env.step(fire_action)
             # 可能需要多執行幾步 FIRE？或者渲染 FIRE 後的畫面
             rendered_frame = env.render() # 渲染 FIRE 之後的畫面
             frames.append(rendered_frame)
             state = preprocessor.step(fire_obs) # 更新狀態堆疊

        while not done:
            # --- 渲染當前畫面 (在動作選擇之前) ---
            # 由於 frame skip，這裡渲染的是上一個 frame skip 週期的最後一幀對應的畫面
            # 但為了影片流暢，我們在 frame skip 內部渲染每一幀
            # rendered_frame = env.render()
            # frames.append(rendered_frame)
            # ---------------------------------

            # --- 選擇動作 ---
            state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device) # <<< 確保類型為 float32 >>>
            with torch.no_grad():
                # <<< 使用 get_expected_q_values 計算期望 Q 值 >>>
                expected_q = model.get_expected_q_values(state_tensor) # Shape: (1, num_actions)
                action = expected_q.argmax().item()
            # -----------------

            # --- Frame Skip 邏輯 ---
            accumulated_reward_fs = 0.0
            last_obs = obs # 保留 frame skip 開始前的 obs
            for _ in range(frame_skip):
                next_obs, reward, terminated, truncated, info = env.step(action)
                accumulated_reward_fs += reward
                done = terminated or truncated

                # <<< 在 frame skip 內部渲染每一幀以生成影片 >>>
                rendered_frame = env.render()
                frames.append(rendered_frame)
                # ------------------------------------------

                last_obs = next_obs # 更新最後觀察到的畫面
                episode_steps += 1
                if done:
                    break # 如果 episode 在 frame skip 中結束，則跳出
            # -----------------------

            total_reward += accumulated_reward_fs

            # 使用 frame skip 週期中的最後一個觀察來更新狀態堆疊
            if last_obs is not None: # 確保 last_obs 有效
                state = preprocessor.step(last_obs)
            # else: # 理論上不應該發生，除非 episode 在第一步就結束
                # state = preprocessor.step(obs) # 或者保持原狀態

        # --- 儲存影片 ---
        if frames: # 確保有幀可以儲存
            out_path = os.path.join(args.output_dir, f"{args.env_name.replace('/', '_')}_ep{ep}_fs{frame_skip}_rew{total_reward:.0f}.mp4")
            try:
                with imageio.get_writer(out_path, fps=args.fps) as video: # <<< 使用參數 fps >>>
                    for f in frames:
                        video.append_data(f)
                print(f"Saved episode {ep} with total reward {total_reward} ({episode_steps} steps) → {out_path}")
            except Exception as e:
                print(f"Error saving video for episode {ep}: {e}")
        else:
            print(f"Episode {ep} ended with total reward {total_reward} ({episode_steps} steps), but no frames were captured.")
        # ---------------
        all_rewards += total_reward

    avg_reward = all_rewards / args.episodes
    print(f"\nAverage reward over {args.episodes} episodes: {avg_reward:.2f}")
    env.close()
    # cv2.destroyAllWindows() # 通常不需要手動調用

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Dueling C51 DQN model for Atari.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained DuelingC51DQN .pt model")
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5", help="Environment ID")
    parser.add_argument("--output-dir", type=str, default="./eval_videos", help="Directory to save evaluation videos")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run for evaluation") # 預設減少，評估通常不需要太多
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")

    # <<< 新增 FrameSkip 參數 >>>
    parser.add_argument("--frame-skip", type=int, default=4, help="Number of frames to skip per action decision")

    # <<< 新增 C51 參數 (必須與訓練時的模型匹配) >>>
    parser.add_argument("--num-atoms", type=int, default=51, help="Number of atoms used for the C51 model being loaded")
    parser.add_argument("--vmin", type=float, default=-4.0, help="Minimum value of the C51 support for the loaded model")
    parser.add_argument("--vmax", type=float, default=4.0, help="Maximum value of the C51 support for the loaded model")

    # <<< 新增影片 FPS 參數 >>>
    parser.add_argument("--fps", type=int, default=30, help="FPS for the output evaluation video")

    args = parser.parse_args()

    # --- Vmin/Vmax 驗證 ---
    if args.vmin >= args.vmax:
        raise ValueError("--vmin must be strictly less than --vmax")
    # --------------------

    evaluate(args)