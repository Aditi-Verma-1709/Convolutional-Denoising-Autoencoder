# Denoising Autoencoder on STL-10 Images

##  Objective
This project implements a **Denoising Autoencoder (DAE)** to restore noisy grayscale images using deep learning. The autoencoder is trained to reconstruct clean versions of images corrupted by Gaussian noise.

##  Dataset
We use the **STL-10** dataset, which contains natural images of 96x96 resolution. For this experiment:
- A subset of **100 grayscale images** is selected for training.
- **20 images** are used for testing.
- All images are resized to **64x64** and converted to **grayscale**.
  
##  Noise Injection
To simulate real-world corruption:
- We apply **Gaussian noise** to a fixed number of random pixels in each image.
- Both a **loop-based** and a **vectorized** noise function are used and compared for efficiency.

## Model: Denoising Autoencoder
A **convolutional autoencoder** is built using PyTorch. It consists of:
- An **encoder** that compresses the noisy image into a latent representation.
- A **decoder** that reconstructs the clean image from this latent space.

## Training Strategy
- **K-Fold Cross-Validation** (k=5) is used for reliable evaluation.
- We perform **Grid Search** over learning rate, batch size, and epochs to identify the best hyperparameters.
- The model is evaluated using:
  - **MSE (Mean Squared Error)**
  - **PSNR (Peak Signal-to-Noise Ratio)**

##  Performance Profiling
The PyTorch **Profiler** is used to analyze:
- Time spent on different operations (CPU/CUDA)
- Memory usage
- FLOPs and throughput in GFLOPs/sec

##  Visualization
Finally, reconstructed images are plotted alongside their noisy inputs and ground truths to visually assess denoising quality.
