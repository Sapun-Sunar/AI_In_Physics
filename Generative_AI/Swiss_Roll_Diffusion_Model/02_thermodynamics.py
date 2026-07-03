import torch
from sklearn.datasets import make_swiss_roll

x0_np, _ = make_swiss_roll(n_samples=3000, noise=0.1)
x0 = torch.tensor(x0_np[:,[0,2]],dtype=torch.float32)
x0 = (x0-x0.mean(dim=0))/x0.std(dim=0)

T=100
beta = torch.linspace(0.001,0.5,T)
alpha = 1 - beta
alpha_cum = torch.cumprod(alpha, dim=0)