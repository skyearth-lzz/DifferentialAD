"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-02-19
@Description：calculate attention
==================================================
"""
import math
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.rms_normalization import RMSNorm


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, - np.inf)
        attn = scale * scores

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        # sigma: [B H L] -> [B H L 1] -> [B H L L]
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attn, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )

        # [B L H d] -> [B L D]
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class DifferentialAttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, depth, window_size, d_keys=None, attention_dropout=0.0,
                 d_values=None):
        super(DifferentialAttentionLayer, self).__init__()
        self.num_heads = n_heads
        self.num_kv_heads = n_heads // 2
        self.head_dim = d_model // n_heads // 2
        self.scaling = self.head_dim ** -0.5
        self.window_size = window_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model // 2, bias=False)
        self.v_proj = nn.Linear(d_model, d_model // 2, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(attention_dropout)

        self.sigma_projection = nn.Linear(d_model, n_heads)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
        self.distances = torch.zeros((window_size, window_size)).cuda()
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)


    def forward(self, queries, keys, values, attn_mask):
        batch_size, length, d_model = queries.size()
        x = queries

        q = self.q_proj(queries)
        k = self.k_proj(keys)
        v = self.v_proj(values)
        sigma = self.sigma_projection(x)

        q = q.view(batch_size, length, 2 * self.num_heads, self.head_dim)
        k = k.view(batch_size, length, 2 * self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, length, self.num_kv_heads, 2 * self.head_dim)
        sigma = sigma.view(batch_size, length, self.num_heads)

        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), 2)
        v = repeat_kv(v.transpose(1, 2), 2)

        scores = torch.matmul(q, k.transpose(-1, -2))
        if attn_mask:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(batch_size, length, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attention_weights = self.scaling * scores
        attention_weights = F.softmax(attention_weights, dim=-1, dtype=torch.float32).type_as(
            attention_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attention_weights = attention_weights.view(batch_size, self.num_heads, 2, length, length)
        # attention_weights: [B, 2H, 2, L, L]
        attention_weights = attention_weights[:, :, 0] - lambda_full * attention_weights[:, :, 1]

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        # sigma: [B H L] -> [B H L 1] -> [B H L L]
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, self.window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(torch.softmax(attention_weights, dim=-1))
        V = self.subln(torch.matmul(attention_weights, v)) * (1 - self.lambda_init)
        out = V.transpose(1, 2).reshape(batch_size, length, self.num_heads * self.head_dim * 2)

        return self.out_proj(out), series, prior, sigma






