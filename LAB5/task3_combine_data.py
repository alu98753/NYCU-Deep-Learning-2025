import numpy as np
import random
import os

file_a = 'expert_pong_score20plus.npz'
file_b = 'expert_data_filteredB.npz'
output_file = 'expert_pong_score21plus.npz' # 最終文件名

try:
    data_a = np.load(file_a)
    data_b = np.load(file_b)

    print(f"Loaded {len(data_a['states'])} transitions from {file_a}")
    print(f"Loaded {len(data_b['states'])} transitions from {file_b}")

    # 合併數據
    combined = {}
    for key in data_a.files:
        if key in data_b:
            combined[key] = np.concatenate((data_a[key], data_b[key]), axis=0)
        else:
            print(f"Warning: Key {key} not found in {file_b}")
            combined[key] = data_a[key] # 或者報錯

    total_combined = len(combined['states'])
    print(f"Total combined transitions: {total_combined}")

    # 打亂數據
    indices = np.arange(total_combined)
    np.random.shuffle(indices)

    shuffled_combined = {}
    for key in combined:
        shuffled_combined[key] = combined[key][indices]

    # 保存合併後的數據
    np.savez_compressed(output_file, **shuffled_combined)
    print(f"Saved combined and shuffled data to {output_file}")

    # 可選：刪除臨時文件
    # import os
    os.remove(file_a)
    os.remove(file_b)

except Exception as e:
    print(f"Error combining data: {e}")