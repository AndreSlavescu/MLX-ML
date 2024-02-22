import mlx.core as mx
import mlx.nn as nn
import numpy as np

# plotting
import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 ** 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 ** 2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

if __name__ == "__main__":
    autoencoder = AutoEncoder()

    a = mx.random.uniform(shape=[784])
    b = autoencoder.forward(a)

    np_a = np.array(mx.reshape(a, [28, 28]))
    np_b = np.array(mx.reshape(b, [28, 28]))

    plt.figure(figsize=(10, 5))  
    plt.subplot(1, 2, 1) 
    plt.title("Base Image")
    plt.imshow(np_a, cmap='gray') 
    plt.colorbar() 

    plt.subplot(1, 2, 2) 
    plt.title("Predicted Image")
    plt.imshow(np_b, cmap='gray') 
    plt.colorbar()

    plt.show()
