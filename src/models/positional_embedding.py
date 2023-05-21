import math
import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


if __name__ == '__main__':
    step_embeder = SinusoidalPosEmb(dim=256)
    timesteps = torch.arange(100, dtype=torch.long, device='cpu')
    x = timesteps.detach().numpy()
    timesteps = timesteps.expand(timesteps.shape[0])
    step_embedding = step_embeder(timesteps)
    y = step_embedding.squeeze().detach().numpy()
    import matplotlib.pyplot as plt
    plt.grid(True)
    plt.plot(x, y[:, 50:100])
    plt.show()
    print(step_embedding.shape)
