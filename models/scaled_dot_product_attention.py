import mlx.core as mx
import mlx.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        import math
        self.sqrt_dim = math.sqrt(dim)

    def forward(self, query, key, value): # not including masking
        score = mx.matmul(query, key.transpose()) / self.sqrt_dim 
        attention = mx.matmul(mx.softmax(score, -1), value) # attention: ((QK^T) / sqrt(d)) * V

        return attention


if __name__ == "__main__":
    dim = 1024
    sequence_length = 10
    scaled_dot_product_attention = ScaledDotProductAttention(dim)

    query = mx.random.uniform(low=0, high=30000, shape=[sequence_length, dim])
    key = mx.random.uniform(low=0, high=30000, shape=[sequence_length, dim])
    value = mx.random.uniform(low=0, high=30000, shape=[sequence_length, dim])

    result = scaled_dot_product_attention.forward(query=query, key=key, value=value) 
    print(result)