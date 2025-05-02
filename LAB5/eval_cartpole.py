import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import argparse

class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):
        return self.network(x)

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1", render_mode="human" if args.render else None)
    num_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    model = DQN(state_dim, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    all_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_reward = 0
        done = False
        while not done:
            state = torch.from_numpy(np.array(obs)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state).argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        all_rewards.append(total_reward)
        print(f"Episode {ep} Reward: {total_reward}")

    print(f"\n Average Reward over {args.episodes} episodes: {np.mean(all_rewards):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to model .pt file")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true", help="Render the environment")
    args = parser.parse_args()
    evaluate(args)

'''command to run the script
python eval_cartpole.py --model-path /home/clu98753cs13/Desktop/DL/LAB5/results/LAB5_313554044_task1_cartpole.pt
'''