# ShapeVAE: Learning and Generating 2D Shapes

## 📌 Overview
ShapeVAE is a **Variational Autoencoder (VAE) with a Transformer-based architecture** that learns to encode and generate 2D shapes. The model is trained on sequences of 2D points extracted from SVG files, learning to reconstruct paths and generate new, unique shapes.

## 🚀 Features
- **Transformer-based Encoder & Decoder** for processing sequential 2D points.
- **VAE Architecture** enables meaningful latent space interpolation.
- **Scalable Training** to learn from thousands of SVGs.
- **Inference Mode** to generate new shapes and visualize them.

## 🏗️ Model Architecture
1. **Encoder (TransformerEncoder)**
   - Embeds 2D points (x, y) into a higher-dimensional space.
   - Processes sequences with Transformer layers.
   - Outputs mean (`mu`) and log variance (`logvar`) for latent representation.

2. **Latent Space (VAE Reparameterization)**
   - Uses `mu` and `logvar` to sample latent vectors via the reparameterization trick.

3. **Decoder (TransformerDecoder)**
   - Takes the latent vector and reconstructs the sequence of 2D points.
   - Outputs generated paths in a structured format.

## 📊 Training Details
- **Dataset:** SVG files converted into sequences of 2D points.
- **Loss Function:** VAE loss = MSE (Reconstruction Loss) + KL Divergence.
- **Optimizer:** Adam with a learning rate of `1e-4`.
- **Training Data Size:** At least **10,000 SVGs** recommended for viable results.
- **Hardware:** Runs on both **CPU and GPU** (CUDA-enabled if available).

## 🔧 Setup & Usage
### 1️⃣ Install Dependencies
```bash
pip install torch numpy matplotlib
```

### 2️⃣ Train the Model
```bash
python train.py
```

### 3️⃣ Generate New Shapes
```bash
python inference.py
```

## 📈 Results & Visualization
After training, the model can generate diverse 2D shapes. The `inference.py` script plots the generated shapes using Matplotlib.

## 📌 Notes
- If results are poor, increase training data or tune hyperparameters.
- For best results, use at least **10,000+ SVGs**.

## 🛠 Future Improvements
- Support for more complex shape structures (e.g., curves, multiple paths).
- Conditional shape generation (e.g., generate shapes based on categories).
- Improved latent space interpolation for smoother shape morphing.

---
🎨 **ShapeVAE: Learn & Create Unique 2D Forms!** 🚀

