import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from brain import Spectroscopic_Brain
from data_creation import AbsoluteVoidForge



def train():

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA.")
    
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (mps).")

    else:
        device = torch.device("cpu")
        print("No GPU found, defaulting to cpu.")
        


    data_set = AbsoluteVoidForge(num_samples=2000, matrix_size=300)
    train_loader = DataLoader(data_set, batch_size=20, shuffle=True)
    brain = Spectroscopic_Brain().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(brain.parameters(), lr=0.0001)
    
    start_epoch = 0
    try:
        # 1. Load the state of the network.
        checkpoint = torch.load("spectroscopic_brain.pth", map_location=device)
        
        # 2. Extract the memories using standard dictionary keys
        brain.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epochs"]
        
        print(f"Checkpoint found. Resuming from Epoch {start_epoch}.")

    except FileNotFoundError:
        print("No previous model found. Making a new one.")
    


    epochs = 50
    best_loss = float("inf")
    patience_counter = 0
    patience_limit = 5

    for epoch in range(start_epoch, epochs):
        brain.train()
        running_loss = 0
    
        for batch_idx, (images,true_fwhm) in enumerate(train_loader):
            images = images.to(device)
            true_fwhm = true_fwhm.to(device)
            optimizer.zero_grad()

            predictions = brain(images)
            loss = criterion(predictions, true_fwhm)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx%25 == 0:
                print(f"loss at epoch {epoch} is {loss.item():.4f}")
        
        avg_epoch_loss = running_loss/len(train_loader)

        if avg_epoch_loss<best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0

            checkpoint = {
                "epochs":epoch + 1,
                "model_state_dict": brain.state_dict(),  
                "optimizer_state_dict": optimizer.state_dict()
            }
            torch.save(checkpoint, "spectroscopic_brain.pth")
            
        else:
            patience_counter +=1

        if patience_counter == patience_limit:
            print("Convergence point reached.")
            break


if __name__ == "__main__":
    train()                                                                                                       