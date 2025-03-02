"""
=================================================
@Author: Zhenzhou Liu
@Date: 2025-03-01
@Description：
==================================================
"""
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from utils.rms_normalization import RMSNorm


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # 在这里，x的维度从 (bs, n_kv_heads, slen, head_dim) 变为 (bs, n_kv_heads, 1, slen, head_dim)，
        # 然后通过expand扩展到 (bs, n_kv_heads, n_rep, slen, head_dim)，
        # 其中第三维的n_rep维度数据是通过复制原始数据填充的。
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth):
    """计算在编码器每层中的 lambda_{init}

    Args:
        depth: 当前层数

    Returns:
        lambda_init: 当前层的lambda_{init}
    """
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadDiffAttn(nn.Module):
    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0,
            depth=0,
            batch_first=True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = num_heads

        # caculate the constant: $\frac{1}{\sqrt{d}}$
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.batch_first = batch_first

        self.q1_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q2_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k1_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k2_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.out_proj = nn.Linear(2 * embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
    ):
        query_win_len, bsz, embed_dim = query.size()
        key_win_len, bsz, embed_dim = key.size()

        query = query.permute(1, 0, 2)
        key = key.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

        # Project x to get q1, q2, k1, k2, v
        q1 = self.q1_proj(query).view(bsz, query_win_len, self.n_head, self.head_dim)
        q2 = self.q2_proj(query).view(bsz, query_win_len, self.n_head, self.head_dim)
        k1 = self.k1_proj(key).view(bsz, key_win_len, self.n_head, self.head_dim)
        k2 = self.k2_proj(key).view(bsz, key_win_len, self.n_head, self.head_dim)
        v = self.v_proj(value).view(bsz, key_win_len, self.n_head, 2 * self.head_dim)
        if self.batch_first:
            q1 = q1.transpose(1, 2)
            q2 = q2.transpose(1, 2)
            k1 = k1.transpose(1, 2)
            k2 = k2.transpose(1, 2)
            v = v.transpose(1, 2)


        att1 = torch.matmul(q1, k1.transpose(-2, -1)) * self.scaling
        att2 = torch.matmul(q2, k2.transpose(-2, -1)) * self.scaling

        att1 = F.softmax(att1, dim=-1)
        att2 = F.softmax(att2, dim=-1)

        # Compute λ for each head separately
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1)).unsqueeze(-1).unsqueeze(-1)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        att = att1 - lambda_full * att2
        att = self.dropout(att)
        y = torch.matmul(att, v)  # [B, n_head, T, 2 * head_size]
        y = self.subln(y)
        y = y * (1 - self.lambda_init)

        y = y.transpose(1, 2).contiguous().view(bsz, query_win_len, self.n_head * 2 * self.head_dim)
        y = y.permute(1, 0, 2)

        return self.out_proj(y), att
