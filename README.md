# AI_In_Physics: Bare-Metal Neural Architectures for Computational Physics

## The Objective
This repository contains a collection of custom, from-scratch machine learning architectures engineered specifically to solve complex challenges in quantum optics, thermodynamics, and physical telemetry. 

Standard deep learning models are optimized for commercial classification and often suffer mathematical collapse when subjected to the strict laws of physics, sensor degradation, or chaotic noise floors. The architectures within this repository bypass high-level API wrappers, utilizing bare-metal PyTorch and custom physical data-forges to enforce physical laws, achieve sub-Nyquist resolution, and extract fundamental properties from simulated quantum mechanics.

---

## Architectural Taxonomy & Modules

This repository is strictly divided into isolated domains of physics, each conquered by a specialized neural architecture.

### I. The Sub-Pixel Interferometric Spectrometer (Quantum Optics)
**Objective:** Sub-pixel telemetry of quantum absorption features (Lorentzian FWHM) within a cold gas chamber.

* **The Physics:** Models the exact thermodynamic and optical mechanics of a Fabry-Perot interference cavity. A live data-forge generates 250x400 true-to-life 2D Gaussian thermal plasma envelopes, projecting Airy transmission masks and localized Lorentzian absorption craters.

* **The Architecture:** A fully custom Convolutional Neural Network (CNN) built in PyTorch. It utilizes greedy `MaxPool2d` layers to preserve high-voltage optical spikes against extreme Poisson (Quantum Shot) and Gaussian (Sensor Read) noise.
 
* **The Result:** Bypasses standard Nyquist optical sampling limits, accurately resolving localized absorption dips to an accuracy of ~0.2 physical pixels.

### II. Convolution Network Occlusion Theory (Chaotic Data Occlusion)
**Objective:** Robust extraction of the Full Width at Half Maximum (FWHM) from highly degraded 2D interferometry data.
* **The Physics:** Simulates catastrophic physical sensor failure and hardware degradation on an optical ring system.
* **The Architecture:** Utilizes a forced 50% data occlusion rate during the training phase. By violently blacking out random sectors of the plasma ring, the network is stripped of its ability to memorize static background noise, forcing it to learn the fundamental geometry of the absorption line.
* **The Result:** Saliency Map (Visual Cortex) extraction proves that even when the physical ring is shattered by sensor damage, the digital optic nerve successfully traces the ghost of the missing curvature to calculate the FWHM.

### III. DDPM Micro-Engine: The Swiss Roll Manifold (Generative AI)
**Objective:** A from-scratch implementation of a Denoising Diffusion Probabilistic Model (DDPM) to demonstrate absolute mastery over Langevin dynamics and the Score Function $\nabla_x \log p(x)$.
* **The Physics:** Isolates the raw thermodynamics of generative AI. Defines a Markov Chain forward-diffusion process that injects Gaussian entropy over $T=100$ timesteps, mathematically destroying a 2D geometric spiral.
* **The Architecture:** A miniaturized 1D Multi-Layer Perceptron (MLP) utilizing purely linear layers. It acts as the Score Function, learning the exact angle and magnitude required to push random static toward a high-density probability manifold.
* **The Result:** Operates completely independently during inference, hallucinating a pristine, novel geometric spiral out of pure mathematical static by mapping the "downhill" gravity of the Latent Space.

---

## Foundational Literature & Physics Constants
The architectural methodology and physical constants utilized across these engines build upon the following frameworks:
* **Deep Learning:** Goodfellow, I., Bengio, Y., & Courville, A. (2016). *MIT Press*.
* **Neural Regularization:** DeVries, T., & Taylor, G. W. (2017). *Improved Regularization of Convolutional Neural Networks with Cutout*.
* **Optics & Wave Mechanics:** Hecht, E. (2016). *Optics (5th ed.)*. Pearson.
* **Computational Physics:** Carleo, G., et al. (2019). *Machine learning and the physical sciences*. Reviews of Modern Physics.

---

## Future Directives & Phase II 
**[!] STATUS: Active Campaign.** While the foundational manifolds established here are mathematically indestructible, this repository is actively evolving for high-energy deployments. 
* **Physics-Informed Neural Networks (PINNs):** Upgrading the loss criteria to strictly enforce physical laws via differential equations during the backpropagation loops.
* **High-Energy Particle Physics Migration:** Transitioning from optical thermodynamics in Python/PyTorch to petabyte-scale high-energy event reconstruction utilizing bare-metal C++ and the CERN ROOT framework for $Z$ and $W$ Boson momentum scale calibrations.

> *"Generative models do not memorize data points; they memorize the gravity of the data."* > — S. Sunar