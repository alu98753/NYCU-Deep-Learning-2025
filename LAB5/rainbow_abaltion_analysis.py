import wandb
import pandas as pd
import os
import time # 增加 time 模塊

# --- 配置 ---
# !!! 將下面兩行替換成你自己的 W&B Entity 和 Project !!!
WANDB_ENTITY = "alu98753-national-yang-ming-chiao-tung-university"
WANDB_PROJECT = "uncategorized" # 或者 "DLP-Lab5-DuelingC51-Pong" 等你的項目名稱

# 輸出文件名
OUTPUT_CSV_FILE = "all_runs_detailed_history.csv"

import wandb

api = wandb.Api()

# 換成你的帳號與專案名
runs = api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}")

for run in runs:
    print(f"Run: {run.name}")
    print("Config:")
    for key, val in run.config.items():
        print(f"  {key}: {val}")
    print("Final reward:", run.summary.get("Evaluation/Raw Reward"))
    print("===")
import pandas as pd

data = []
for run in runs:
    cfg = run.config
    row = {**cfg}
    row["run_id"] = run.id
    row["eval_reward"] = run.summary.get("Evaluation/Raw Reward")
    data.append(row)

df = pd.DataFrame(data)
df.to_csv("wandb_run_summary.csv", index=False)
