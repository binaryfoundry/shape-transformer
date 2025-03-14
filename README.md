# Shape Transformer Variational Autoencoder

This project implements a **Transformer-based Variational Autoencoder (VAE)** for learning and interpolating 2D shapes. The model encodes shapes into a structured latent space using a Transformer, enabling smooth shape reconstruction and interpolation with better generalization.

## Model Architecture
The model consists of three main components:

### 1. **Encoder**
- Maps 2D points to a **latent distribution** (mean **μ** and log-variance **logσ²**) using a linear layer.
- Uses the **reparameterization trick** to sample latent vectors, allowing stochastic encoding.

### 2. **Transformer**
- Processes latent representations to **capture spatial relationships**.
- Uses **multi-head self-attention** for improved feature extraction and robustness to transformations.

### 3. **Decoder**
- Maps the sampled latent vector **back to 2D points** for shape reconstruction.

## Why Use a Variational Autoencoder (VAE) and Latent Space?

### **Structured Latent Representations**
- Unlike standard autoencoders, VAEs learn a **probabilistic latent space**, ensuring smooth transitions between shapes and preventing overfitting to specific examples.

### **Better Interpolation and Robustness**
- Traditional autoencoders learn absolute vertex positions, making interpolations sensitive to shape order or transformations (e.g., flipping).
- VAEs enforce **smooth and meaningful latent transitions**, making interpolation **more stable and natural**.

### **Regularization with KL Divergence**
- The **Kullback-Leibler (KL) divergence loss** ensures that the latent space follows a **continuous and structured Gaussian distribution**, leading to better generalization to unseen shapes.

## Training
The model is trained using a **combined loss function**:
1. **Reconstruction Loss** (Mean Squared Error - MSE): Measures how well the reconstructed shape matches the input.
2. **KL Divergence Loss**: Regularizes the latent space to follow a Gaussian distribution.

## Example: Shape Interpolation
By encoding a **square** and a **circle** into the latent space, we can smoothly interpolate between them, generating **intermediate shapes** that blend both features.

## Future Improvements
- Implementing **latent space arithmetic** (e.g., adding geometric features).
- Experimenting with **higher-dimensional latent spaces** for more complex shapes.
- Exploring **conditional VAEs** to generate shapes based on user-defined attributes.
