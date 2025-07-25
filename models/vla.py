import torch.nn as nn
from einops import rearrange
import torch
import torch.nn.functional as F
from typing import List, Optional
from torch import Tensor

class MMF(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt * tgt2
        return tgt

class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q_x = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_memory = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, focusing_factor, kernel_size))

    def forward(self, x, memory, H, W, T, l):
        x = x.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)
        B, _, C = x.shape
        N_x = H * W * T * l
        N_memory = H * W * T

        q_x = self.q_x(x)
        k_memory = self.k_memory(memory)
        v = self.v(memory)

        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)

        def process_tensor(tensor):
            tensor = kernel_function(tensor) + 1e-6
            tensor = tensor / scale
            tensor_norm = tensor.norm(dim=-1, keepdim=True)
            tensor = tensor ** focusing_factor
            tensor = (tensor / tensor.norm(dim=-1, keepdim=True)) * tensor_norm
            return tensor

        q_x = process_tensor(q_x)
        k_memory = process_tensor(k_memory)

        q_x = q_x.reshape(B, N_x, self.num_heads, -1).permute(0, 2, 1, 3)
        k_memory = k_memory.reshape(B, N_memory, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N_memory, self.num_heads, -1).permute(0, 2, 1, 3)

        z_x = 1 / (q_x @ k_memory.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_memory.transpose(-2, -1) * (N_memory ** -0.5)) @ (v * (N_memory ** -0.5))
        x = q_x @ kv * z_x

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, N_memory), size=N_x,
                                          mode='linear').reshape(B, self.num_heads, -1, N_x).transpose(-2, -1)
        x = x.transpose(1, 2).reshape(B, N_x, C)
        v = v.reshape(B * self.num_heads, H, W, T, -1).permute(0, 4, 1, 2, 3)
        x = x + self.dwc(v.flatten(start_dim=3)).reshape(B, C, N_memory).permute(0, 2, 1).repeat(1, l, 1)

        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.permute(1, 0, 2)

        return x
    
class LinearCrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert (
            self.head_dim * nhead == self.d_model
        ), "d_model must be divisible by nhead"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        query, key, value = tgt, memory, memory
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        q = q.view(q.size(0), q.size(1), self.nhead, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.nhead, self.head_dim).transpose(1, 2)

        k_prime = torch.nn.functional.elu(k) + 1
        q_prime = torch.nn.functional.elu(q) + 1
        context = torch.einsum('bhnd,bhne->bhde', k_prime, v)
        attn_output = torch.einsum('bhnd,bhde->bhne', q_prime, context)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            attn_output.size(0), attn_output.size(2), self.d_model
        )

        attn_output = self.out_proj(attn_output).permute(1,0,2)
        return attn_output