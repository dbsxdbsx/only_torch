"""
MultiHeadAttention Python 参考实现（PyTorch 对照值）

用于验证 only_torch 的 MultiHeadAttention 实现正确性。
"""

import torch
import torch.nn as nn
import numpy as np

def attention_reference():
    """MultiHeadAttention 对照值"""
    torch.manual_seed(42)

    embed_dim = 8
    num_heads = 2
    seq_len = 3
    batch_size = 2

    # 创建 MultiHeadAttention
    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    # 输入: [batch, seq_len, embed_dim]
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    print("=== MultiHeadAttention ===")
    print(f"embed_dim={embed_dim}, num_heads={num_heads}")
    print(f"query shape: {query.shape}")
    print(f"key shape: {key.shape}")
    print(f"value shape: {value.shape}")

    # 前向计算
    output, attn_weights = mha(query, key, value)
    print(f"\n输出 shape: {output.shape}")
    print(f"注意力权重 shape: {attn_weights.shape}")
    print(f"输出[0,0,:4]: {output[0, 0, :4].detach().numpy()}")

def scaled_dot_product_reference():
    """Scaled Dot-Product Attention 纯手工实现"""
    torch.manual_seed(42)

    d_k = 4
    seq_len = 3

    Q = torch.randn(1, seq_len, d_k)
    K = torch.randn(1, seq_len, d_k)
    V = torch.randn(1, seq_len, d_k)

    # scores = Q * K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)

    print("\n=== Scaled Dot-Product Attention (手工) ===")
    print(f"Q shape: {Q.shape}")
    print(f"scores:\n{scores[0].detach().numpy()}")
    print(f"attn_weights:\n{attn_weights[0].detach().numpy()}")
    print(f"output:\n{output[0].detach().numpy()}")

if __name__ == "__main__":
    attention_reference()
    scaled_dot_product_reference()
