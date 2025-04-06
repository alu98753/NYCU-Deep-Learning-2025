import torch.nn as nn
import torch
import math
import torch.nn.functional as F

#TODO1
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1] # dim of head
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention # 注意力處理後的輸出（後面會 concat 多個 head 的這個） , 原始的注意力矩陣（可視化觀察模型關注什麼）

class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim//self.num_heads
        self.scale = self.head_dim**0.5
        
        self.qkv_layer =  nn.Linear(dim, 3 * self.num_heads * self.head_dim,bias = False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)


    def forward(self, x,mask=None):
        ''' Hint: input x tensor shape is (batch_size, num_image_tokens, dim), 
            because the bidirectional transformer first will embed each token to dim dimension, 
            and then pass to n_layers of encoders consist of Multi-Head Attention and MLP. 
            # of head set 16
            Total d_k , d_v set to 768
            d_k , d_v for one head will be 768//16.
        '''
        batch_size, num_image_tokens, dim  = x.size()
        # print(f"x.size():,{x.size()}")
        qkv = self.qkv_layer(x)
        # print(f"qkv.size():,{qkv.size()}")
        qkv = qkv.reshape(batch_size, num_image_tokens , self.num_heads ,3 * self.head_dim)
        # print(f"qkv.size():,{qkv.size()}")
        qkv = qkv.permute(0, 2, 1, 3)
        # print(f"qkv.size():,{qkv.size()}")
        q,k,v = qkv.chunk(3, dim=-1)
        
        # print(f"q size: {q.size()}, k size: {k.size()}, v size: {v.size()}, ")
        values, attention = scaled_dot_product(q, k, v, mask)
        # print(f"values.size(): {values.size()}, attention.size:{ attention.size()} ")
        values = values.reshape(batch_size, num_image_tokens, self.num_heads * self.head_dim)
        # print(f"values.size(): {values.size()}")
        out = self.out_proj(values)
        # print(f"out.size(): {out.size()}")
        return out

# 測試
# batch_size = 2
# seq_len = 10
# embed_dim = 768
# num_heads = 16

# x = torch.randn(batch_size, seq_len, embed_dim)
# mha = MultiHeadAttention(dim=embed_dim, num_heads=num_heads)
# output = mha(x)

# # 輸出形狀檢查
# print("Input shape:", x.shape)       # [2, 10, 768]
# print("Output shape:", output.shape) # [2, 10, 768]

class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )
        
    def forward(self, input):
        return super().forward(input)
    
    
class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)
        
        x = x + attn
        x = self.LayerNorm1(x)
        
        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)
    