import torch 
import torch.nn as nn
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        # 1. 通過 VQGAN encoder 得到特徵向量與離散化結果
        '''
        codebook_mapping: [B, C, H, W] → 對應量化後的特徵（z_q_mapping）
        codebook_indices: [B, H, W] → 對應 codebook 向量的索引（z_indices）
        q_loss: quantization loss（訓練時才會用）
        '''
        
        z_q_mapping , z_indices , _ = self.vqgan.encode(x)
        # 2. 將 z_indices reshape 成平坦的 token 序列
        z_indices = z_indices.reshape(z_q_mapping.shape[0], -1)  # [B, H*W]
        # print(z_indices.shape)   #  [10, num_tokens]
        # print(z_q_mapping.shape) #  [1, C, H, W]
        
        return  z_q_mapping ,z_indices
    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda r:1-r
        elif mode == "cosine":
            return lambda r: np.cos((r * np.pi) /2 )
        elif mode == "square":
            return lambda r: 1- r**2
        else:
            raise NotImplementedError(f"Unknown gamma mode: {mode}")

##TODO2 step1-3:            
    def forward(self, x):
        # 1. 將圖片轉成 token index（z_indices: ground truth） 
        _, z_indices = self.encode_to_z(x)  # [B, T]，T = num_image_tokens

        B, T = z_indices.shape
        device = z_indices.device

        # 2. 動態產生每個 sample 的遮罩比例
        random_ratios = torch.rand(B, device=device)
        mask_ratios = torch.tensor([self.gamma(r.item()) for r in random_ratios], device=device)  # [B]
        num_mask = (mask_ratios * T).long()  # 每個 sample 遮住的 token 數 [B]

        # 3. 為每個 sample 隨機產生排序索引
        rand_score = torch.rand(B, T, device=device)         # 每個 token 的亂數分數
        sorted_idx = rand_score.argsort(dim=1)               # 排序後的 token 位置

        # 4. 建立遮罩：前 num_mask 個位置為 True，其餘為 False
        mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        row_indices = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
        col_indices = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
        col_mask = col_indices < num_mask.unsqueeze(1)             # [B, T]
        mask.scatter_(1, sorted_idx, col_mask)                     # 在排序後的位置上設 True

        # 5. 用 mask_token_id 替換被遮住的位置
        input_indices = torch.where(mask, torch.full_like(z_indices, self.mask_token_id), z_indices)

        # 6. 傳入 Transformer，預測 logits
        # z_indices=None #ground truth
        logits = self.transformer(input_indices) #transformer predict the probability of tokens

        return logits, z_indices
        

    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, z_indices, mask_func, mask_num, mask, ratio):
        # print(mask.shape) # [1,256]
        # print(self.mask_token_id) # 1024
        # print(z_indices.shape) # [1,256]
        #  將 mask 的位置替換為 mask token
        num = mask.sum()
        z_indices_mask =  torch.where(mask, torch.full_like(z_indices, self.mask_token_id), z_indices)
        logits = self.transformer(z_indices_mask) # logits with z_indices with mask
        probs = torch.softmax(logits, dim=-1)
        
        # 從 logits 中 sample（避免 sample 到 mask_token_id）
        z_indices_predict = torch.distributions.categorical.Categorical(logits=logits).sample()
        while torch.any(z_indices_predict == self.mask_token_id):
            z_indices_predict = torch.distributions.categorical.Categorical(logits=logits).sample()
        
        # 用預測值更新 masked 的地方用新 token，其他保留原本
        z_indices_predict =  torch.where(mask, z_indices_predict, z_indices)

        # 取得每個預測 token 對應的機率 confidence
        z_indices_predict_prob = probs.gather(-1, z_indices_predict.unsqueeze(-1)).squeeze(-1)
        
        # 只對 masked 的地方保留 confidence，其他設為 +∞ 避免被選進排序
        z_indices_predict_prob = torch.where(
            mask, z_indices_predict_prob, torch.zeros_like(z_indices_predict_prob) + torch.inf
            )

        # 當前 mask ratio
        mask_ratio = self.gamma_func(mask_func)(ratio)
                
        # gumbel noise
        g = torch.distributions.gumbel.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(z_indices_predict_prob.device)
        temperature = self.choice_temperature * (1 - mask_ratio)
        confidence = z_indices_predict_prob + temperature * g
        
        #hint: If mask is False, the probability should be set to infinity, so that the tokens are not affected by the transformer's prediction
        #sort the confidence for the rank 
        #define how much the iteration remain predicted tokens by mask scheduling
        ##At the end of the decoding process, add back the original(non-masked) token values
        
        # 根據 mask_ratio 決定保留幾個 mask
        sorted_confidence = torch.sort(confidence, dim=-1)[0]
        
        mask_len = torch.floor(num * mask_ratio).long()

        # 產生新的 mask：保留最不確定的 mask_len 個，其餘填掉
        cut_off = sorted_confidence[:, mask_len].unsqueeze(-1)
        new_mask  = confidence < cut_off
        return z_indices_predict, new_mask 

__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    



        
