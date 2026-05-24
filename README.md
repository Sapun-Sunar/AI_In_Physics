# DDPM Micro-Engine: The Swiss Roll Manifold

## Abstract
This repository contains a from-scratch, bare-metal implementation of a Denoising Diffusion Probabilistic Model (DDPM). Rather than relying on massive Convolutional networks or high-level API wrappers, this project isolates the raw physics of generative AI. 

It utilizes a primitive 1D Multi-Layer Perceptron (MLP) to learn the continuous vector field of a 2D coordinate manifold (the Swiss Roll). The objective of this micro-engine is to demonstrate absolute mastery over Langevin dynamics, the Score Function, and the dimensional manipulation of PyTorch tensors before scaling to spatial pixel grids.


-----------------------------------------------------------------

## Core Architecture
The system is strictly divided into isolated modules to separate the thermodynamic physics from the neural architecture, and the training factory from the generation environment.

### 1. 'Thermodynamics.py' (The Physics Engine)
Defines the Markov Chain forward-diffusion process. 
* Implements the linear noise schedule ($\beta$, $\alpha$).
* Mathematically flattens a 3D coordinate space into a strictly 2D geometric spiral (X-Z projection) for visual telemetry.
* Calculates the exact trajectory of data destruction, injecting Gaussian entropy over $T=100$ timesteps.


### 2. 'Brain.py' (The Neural Matrix)
A miniaturized, highly efficient Multi-Layer Perceptron.
* **Architecture:** Utilizes purely linear layers ('nn.Linear') constrained to 128 hidden dimensions.
* **Input Fusion:** Dynamically concatenates ('torch.cat', 'dim=1') 2D spatial coordinates with normalized temporal embeddings.
* **Output:** Predicts the exact $X$ and $Z$ vectors of the injected noise, effectively mapping the "downhill" gradient of the Latent Space.


### 3. 'Forge.py' (The Factory Floor)
The isolated training loop and calculus engine.
* Initializes the Adam Optimizer and the 'MSELoss' criterion.
* Executes continuous Backpropagation ('loss.backward()') across randomized batches of noisy coordinates.
* Implements persistent cryogenic saving, serializing the resulting '.pth' weights to the local drive to prevent catastrophic forgetting between sessions.


### 4. 'Conjure.py' (The Exhibition)
The reverse-diffusion generation script.
* Operates completely independently of the Forge ('torch.no_grad()') for maximum CPU/RAM efficiency.
* Spawns 3,000 particles of pure mathematical static from the void and utilizes the trained matrix to guide them downhill along the learned probability distribution, hallucinating a pristine, novel geometric spiral.


------------------------------------------------------------------


## Mathematical Foundations
This engine proves that generative models do not memorize data points; they memorize the *gravity* of the data. 
* **The Vector Field:** The network acts as a Score Function ($\nabla_x \log p(x)$), learning the exact angle and magnitude required to push random static toward a high-density probability manifold.
* **Dimensional Precision:** Explicitly handles 'dim=0' (Batch standardization) versus 'dim=1' (Feature concatenation) to maintain tensor integrity during multi-variable calculus.


------------------------------------------------------------------


## Execution Protocol

**1. Initialize the Environment**
Ensure PyTorch and Matplotlib are installed in your local environment.
```bash
pip install torch matplotlib scikit-learn  
```

**2. Ignite the Forge (Training)**
To begin training the matrix and carving the vector field:Bash python3 forge.py
Telemetry will output to the terminal. 

Once the epoch is reached, trained_demon_weights.pth will be generated.3. Conjure Reality (Generation) to hallucinate a novel geometric structure from the void:Bash python3 conjure.p
A Matplotlib visualizer will render the finalized 2D geometry. Phase II (The Ascension Roadmap).

**3. Future Directions**
This repository serves as the mathematical foundation for an upcoming architectural overhaul:Spatial Matrices: Purging the 1D nn.Linear layers and installing a full 2D Convolutional U-Net with Skip Connections to process true pixel grids instead of coordinate points.

The Steering Wheel: Integrating a Cross-Attention Mechanism ($Q, K, V$) to physically warp the latent space gravity based on textual prompts. Physics-Informed Loss (PINN): Upgrading the Judge to strictly enforce physical laws via differential equations during the backpropagation loop.