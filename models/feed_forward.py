import mlx.core as mx
import mlx.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.weight_1 = nn.Linear(d_model, d_ff)
        self.weight_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.weight_2(self.dropout(self.relu(self.weight_1(x))))
    

if __name__ == "__main__":
    feed_forward_layer = FeedForward(1, 5)

    a = mx.random.randint(low=0, high=100, shape=[5, 1])
    b = feed_forward_layer.forward(a)
    
    print(b)

    
