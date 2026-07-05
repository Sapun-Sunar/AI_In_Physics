# Convolution Network Occlusion Theory: Robust Absorption Line Extraction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20Ready-EE4C2C)
![License](https://img.shields.io/badge/License-MIT-green)

## The Physics Objective
This repository deploys a custom Convolutional Neural Network (CNN) to extract the Full Width at Half Maximum (FWHM) from noisy 2D Fabry-Perot interferometry data. 

Standard deep learning models trained on perfect laboratory simulations inherently overfit to background thermal noise. When physical sensors fail or degrade, these traditional algorithms experience mathematical collapse. 

This architecture utilizes a forced **50% data occlusion rate** during the training phase. By violently blacking out random sectors of the plasma ring during tensor generation, we create a strict information bottleneck. This strips away the network's ability to memorize static, forcing it to learn the fundamental, underlying geometry of the absorption line.


## The Visual Cortex (Saliency Map)
By calculating the mathematical gradient of the network's final prediction backward through the convolutional layers, we generate a Saliency Heatmap. This proves the network is not blindly guessing. Even when the physical ring is shattered by sensor damage, the digital optic nerve traces the ghost of the missing curvature to calculate the FWHM.


Pristine Verse.
![Saliency Brain Scan](assets/Visual_Brain_Scan1.png)


Occluded Verse.
![Saliency Brain Scan](assets/Visual_Brain_Scan2.png)

*(Note: Saliency overlays comparing the Pristine Verse and Occluded Verse demonstrate the network's active logic mapping).*

## Architectural Taxonomy
The system is constructed as a modular, sequentially numbered pipeline:

* `01_brain.py` (The Anatomy): Contains the `Spectroscopic_Brain` class. A continuous regression CNN utilizing aggressive 5x5 Max Pooling layers to downsample noise while preserving massive geometric curvatures.
* `02_data_creation.py` (The Forge): Dynamically generates thousands of Fabry-Perot matrices using natural inverse-square light decay. Controls the quantum coin-flip for the 50/50 occlusion training dataset.
* `03_train.py` (The Crucible): The training loop that forces the network to converge on the chaotic dataset.
* `04_sip_decoder.py` & `05_find_width.py`: Telemetry scripts designed to interpret and map the spatial coordinates of the tensor outputs.
* `06_arbitration.py` (The Judge): Provides the error percentage made by a network, confirming active physical learning versus static memorization.
* `07_saliencymap.py` (The Photographer): Rips open the AI's visual cortex during inference to generate the thermal overlay of its active logic.

## Foundational Literature
The architectural methodology and physical constants utilized in this engine build upon the following frameworks:
* Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
* DeVries, T., & Taylor, G. W. (2017). *Improved Regularization of Convolutional Neural Networks with Cutout*. arXiv:1708.04552.
* Hecht, E. (2016). *Optics (5th ed.)*. Pearson.
* Carleo, G., et al. (2019). *Machine learning and the physical sciences*. Reviews of Modern Physics, 91(4).

---
*Engineered by Sapun Sunar*