import os
import torch
import matplotlib.pyplot as plt
from Thermodynamics import T, beta, alpha, alpha_cum
from Brain import model


save_path = "trained_demon_weights.pth"

try:
    model.load_state_dict(torch.load(save_path)["model"])
except FileNotFoundError:
    print("No Trained Demon Model Found, Forge One First!")

model.eval()

def generate_universe():
    with torch.no_grad():
        x = torch.randn(3000,2)
        
        for t in reversed(range(1,T)):
            t_tensor = torch.full((3000,),t, dtype=torch.long)

            predicted_noise = model(x,t_tensor)
            alpha_t = alpha[t]
            alpha_t_cum = alpha_cum[t]

            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / (torch.sqrt(1 - alpha_t_cum)) * predicted_noise))

            if t > 1:
                x = x + torch.sqrt(beta[t]) * torch.randn_like(x)

    return x

new_reality = generate_universe()

plt.figure(figsize=(6,6))
plt.scatter(new_reality[:,0], new_reality[:,1], color="cyan", alpha = 0.5, s = 5)
plt.title("Hallucinated Reality")
plt.gca().set_facecolor("black")
plt.show()