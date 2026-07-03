import torch
import torch.nn as nn
import torch.optim as optim
import os

from Thermodynamics import T, alpha_cum, x0
from Brain import model

save_path = "trained_demon_weights.pth"

batch_size = 128
epochs = 2500
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()



if os.path.exists(save_path):
    checkpoint = torch.load(save_path)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1

    print("Trained Demon Model Found, Initializing States!")
else:
    print("No Trained Model Found, Forging New Demon!")
    start_epoch = 0




for epoch in range(start_epoch, start_epoch + epochs):

    indices = torch.randint(0, len(x0), (batch_size,))
    x_batch = x0[indices]
    t_batch = torch.randint(0, T, (batch_size,))

    true_noise = torch.randn_like(x_batch)
    alpha_t = alpha_cum[t_batch].view(-1, 1)

    x_noisy = (
        torch.sqrt(alpha_t) * x_batch +
        torch.sqrt(1 - alpha_t) * true_noise
    )

    optimizer.zero_grad()

    predicted_noise = model(x_noisy, t_batch)

    loss = criterion(predicted_noise, true_noise)

    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"loss at epoch {epoch} = {loss.item():.6f}")


torch.save({
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "epoch": epoch
}, save_path)