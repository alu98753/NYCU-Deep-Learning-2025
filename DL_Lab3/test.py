import torch
print(torch.cuda.is_available())          # True = 成功使用 GPU
print(torch.cuda.get_device_name(0))      # 看看顯示卡型號
