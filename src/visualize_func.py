import torch
import torch.nn as nn
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.grid(True)
    symlog = lambda x: torch.sign(x) * torch.log(1 + torch.abs(x))
    symlog1 = lambda x: 2.0 * torch.sign(x) * torch.log(1 + torch.abs(1.0 * x))
    identity = lambda x: x
    x_in = torch.linspace(-10, 10, 1000)
    y_in = symlog(x_in)
    y_in_1 = symlog1(x_in)
    y_identity = identity(x_in)
    plt.plot(x_in.numpy(), y_in.numpy(), color='red')
    plt.plot(x_in.numpy(), y_in_1.numpy(), color='blue')
    plt.plot(x_in.numpy(), y_identity.numpy(), color='green')
    plt.show()
