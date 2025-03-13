# Shape Transformer Autoencoder

This project implements a **Transformer-based Autoencoder** for learning and interpolating 2D shapes. The model encodes shapes into a latent space using a Transformer, enabling smooth shape reconstruction and interpolation.

## Features
- **Transformer-based Autoencoder** for shape encoding and reconstruction.
- **Latent Space Interpolation** between shapes.
- **Customizable Model Parameters** (embedding size, transformer layers, etc.).

## Model Architecture
The model consists of three main components:

### 1. **Encoder**
- Maps 2D points to a higher-dimensional **latent space** using a linear layer.  

### 2. **Transformer**
- Processes the latent representations to **capture spatial relationships**.  
- Uses **multi-head self-attention** for better feature extraction.  

### 3. **Decoder**
- Maps the processed latent space **back to 2D points** for shape reconstruction.  

## Why Use an Autoencoder and Latent Space?

### **Dimensionality Reduction**
- The autoencoder **compresses** high-dimensional shape information into a compact **latent representation**, allowing for efficient storage and manipulation.

### **Smooth Shape Interpolation**
- By mapping shapes to a continuous latent space, we can interpolate between them smoothly, **generating intermediate shapes** that blend features from both inputs.

### **Feature Extraction & Generalization**
- The Transformer-based encoder **learns meaningful shape representations**, capturing underlying patterns instead of just memorizing input shapes.
- The model can generalize to **unseen shapes** by learning a structured latent space.

## Training
The model is trained using **Mean Squared Error (MSE) Loss** to minimize the reconstruction error between input and output shapes.
