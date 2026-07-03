# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os 
import sys 

from brain import SpectroscopicBrain
from dataset import get_live_dataloaders

def ignite_training(max_epochs=100, initial_learning_rate=0.0001, patience=5):
    """
    The Continuous Feeder and The Multi-Stage Gatekeeper.
    """
    # 1. HARDWARE LINK
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Hardware Link Established: {device} Active.\n")

    # 2. INSTANTIATE THE WEAPONRY
    model = SpectroscopicBrain().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
    
    model_path = "spectroscopic_apex_predator.pth"
    start_epoch = 0
    best_val_loss = float('inf')
    
    # --- THE DEEP STATE RECOVERY & TACTICAL INTERRUPT ---
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        print("\n[?] DEEP STATE DETECTED IN VAULT.")
        print("Are you initiating a new Curriculum Phase? (This preserves the brain, but wipes optimizer momentum and Gatekeeper memory).")
        tactical_command = input("Initiate Curriculum Amnesia? [y/N]: ").strip().lower()
        
        if tactical_command in ['y', 'yes']:
            print("\n[!] CURRICULUM SHIFT CONFIRMED: Executing Gatekeeper Amnesia.")
            print("[*] Neural Weights (Brain): RESTORED.")
            print("[*] Optimizer Momentum: PURGED.")
            print("[*] Gatekeeper History: WIPED.")
            print("Welcome to the next phase, My Lord. Starting from Epoch 0.\n")
        else:
            print("\n[*] SAFE RESUME CONFIRMED: Retrieving prior momentum.")
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            print(f"Deep State perfectly restored. Resuming from Epoch {start_epoch}.\n")
            
    else:
        print("\nNo model found. Forging a mathematically blank intellect.\n")
        
    train_loader, val_loader = get_live_dataloaders(batch_size=32, train_size=2000, val_size=400)
    
    # --- MULTI-STAGE TRANSMISSION VARIABLES ---
    patience_counter = 0
    lr_shift_counter = 0
    MAX_GEAR_SHIFTS = 2 
    
    print(f"Igniting Infinite Generator for max {max_epochs} Epochs. Patience: {patience}\n")
    
    try:
        # 4. THE MASTER LOOP
        for epoch in range(start_epoch, max_epochs):
            
            # --- TRAINING PASS ---
            model.train()
            running_train_loss = 0.0
            
            for batch_matrices, batch_fwhms in train_loader:
                batch_matrices = batch_matrices.to(device)
                batch_fwhms = batch_fwhms.to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                predictions = model(batch_matrices)
                loss = criterion(predictions, batch_fwhms)
                loss.backward()
                optimizer.step()
                
                running_train_loss += loss.item()
                
            avg_train_loss = running_train_loss / len(train_loader)
            
            # --- EVALUATION PASS ---
            model.eval()
            running_val_loss = 0.0
            
            with torch.no_grad():
                for val_matrices, val_fwhms in val_loader:
                    val_matrices = val_matrices.to(device)
                    val_fwhms = val_fwhms.to(device).unsqueeze(1)
                    
                    val_predictions = model(val_matrices)
                    val_loss = criterion(val_predictions, val_fwhms)
                    running_val_loss += val_loss.item()
                    
            avg_val_loss = running_val_loss / len(val_loader)
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1:03d}/{max_epochs:03d}] | LR: {current_lr:.1e} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            # --- THE ADVANCED GATEKEEPER LOGIC ---
            if avg_val_loss < best_val_loss:
                # The Immortal Best is updated
                best_val_loss = avg_val_loss
                patience_counter = 0 
                
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss
                }
                torch.save(checkpoint, model_path)
                print(f"   -> Breakthrough. Deep state secured at {best_val_loss:.6f}.")
            else:
                patience_counter += 1
                print(f"   -> Warning: No improvement. Patience at {patience_counter}/{patience}.")
                
                if patience_counter >= patience:
                    if lr_shift_counter < MAX_GEAR_SHIFTS:
                        lr_shift_counter += 1
                        patience_counter = 0
                        
                        # The Scalpel: Reduce LR by a factor of 10
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.1
                            
                        print(f"\n[!] TRANSMISSION DOWNSHIFT {lr_shift_counter}/{MAX_GEAR_SHIFTS}: Plateaus reached.")
                        print(f"[*] Slashing learning rate to {optimizer.param_groups[0]['lr']:.1e}. Commencing surgical fine-tuning.\n")
                    else:
                        print("\n[!] GATEKEEPER TRIGGERED: Maximum gear shifts exhausted. Overfitting death spiral avoided.")
                        print(f"[!] Terminating training sequence to preserve maximum intellect.")
                        break 
                    
    except KeyboardInterrupt:
        print("\n\n[!] MANUAL OVERRIDE DETECTED (Ctrl+C).")
        print("[!] Initiating emergency vault sequence...")
        
        emergency_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }
        torch.save(emergency_checkpoint, model_path)
        print(f"[*] Emergency Deep State secured at Epoch {epoch}. You may resume the campaign at your leisure.\n")
        sys.exit(0)

    print("\nRetrieving the flawless brain state from the vault...")
    final_checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(final_checkpoint["model_state_dict"])
    model.eval()
    print(f"Spectroscopic Brain is primed and lethal. Final locked Val Loss: {final_checkpoint['best_val_loss']:.6f}")

if __name__ == "__main__":
    ignite_training()