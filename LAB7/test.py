# # from huggingface_hub import snapshot_download

# # # 指定模型的 repo_id
# # repo_id = "crispisu/ppo-Walker2D-v4"

# # # 指定要下載的子資料夾
# # allow_patterns = "ppo-Walker2d-v4/*"

# # # 指定下載的本地目錄
# # local_dir = "./ppo-Walker2d-v4"

# # # 下載指定的資料夾
# # snapshot_download(
# #     repo_id=repo_id,
# #     allow_patterns=allow_patterns,
# #     local_dir=local_dir
# # )


# import torch

# # 指定模型檔案路徑
# model_path = "./ppo-Walker2d-v4/ppo-Walker2d-v4/policy.pth"

# # 嘗試載入模型
# try:
#     checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
#     print("模型成功載入！")
# except Exception as e:
#     print(f"模型載入失敗：{e}")

# # 檢查模型結構
# if isinstance(checkpoint, dict):
#     print("\n檔案內容為字典，包含以下鍵值：")
#     for key in checkpoint.keys():
#         print(f"- {key}")

#     # 若包含 state_dict，提取模型權重
#     if "model" in checkpoint:
#         print("\n模型架構：")
#         print(checkpoint["model"])
#     elif "state_dict" in checkpoint:
#         print("\n模型架構：")
#         print(checkpoint["state_dict"].keys())
#     else:
#         print("\n無法找到模型權重，檢查其他鍵值內容：")
#         for key, value in checkpoint.items():
#             print(f"{key}: {type(value)}")
# else:
#     print("\n模型檔案可能直接是模型本體：")
#     print(checkpoint)

# # 嘗試從檔案中重建模型結構
# try:
#     model = checkpoint['model'] if 'model' in checkpoint else checkpoint
#     print("\n模型架構：")
#     print(model)

#     # 列出模型中的所有層與參數
#     print("\n模型權重名稱與形狀：")
#     for name, param in model.items():
#         print(f"{name}: {param.shape}")
# except Exception as e:
#     print(f"載入模型失敗：{e}")

import torch
import torch.nn as nn

class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(PPOActorCritic, self).__init__()

        # Actor - Policy Network
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.action_net = nn.Linear(64, action_dim)

        # Critic - Value Network
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Log standard deviation (for action distribution)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        # Actor forward pass
        policy_out = self.policy_net(x)
        action_mean = self.action_net(policy_out)
        
        # Action distribution
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, std)
        
        # Critic forward pass
        value = self.value_net(x)

        return dist, value

# 模型參數
obs_dim = 17
action_dim = 6

# 初始化模型
model = PPOActorCritic(obs_dim, action_dim)

# 載入權重
model_path = "./ppo-Walker2d-v4/ppo-Walker2d-v4/policy.pth"
checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

# 處理權重名稱 (去掉 "mlp_extractor." 前綴)
new_state_dict = {}
for k, v in checkpoint.items():
    new_key = k.replace("mlp_extractor.", "")
    new_state_dict[new_key] = v

# 更新模型權重
model.load_state_dict(new_state_dict, strict=False)
print("\n成功還原模型：")
print(model)

# 測試模型
x = torch.randn(1, obs_dim)  # 隨機生成一個觀察
dist, value = model(x)

print("\n動作分佈的均值：", dist.mean)
print("動作分佈的標準差：", dist.stddev)
print("值函數輸出：", value)
