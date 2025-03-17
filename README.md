# DoomGPT: Learning and Generating Doom Maps

## 📌 Overview
DoomGPT is a **Variational Autoencoder (VAE) with a Transformer-based architecture** that learns to encode and generate Doom maps.

## 🚀 Features
- **Transformer-based Encoder & Decoder** for processing sequential 2D points.
- **VAE Architecture** enables meaningful latent space interpolation.

## 🏗️ Model Architecture
1. **Encoder (TransformerEncoder)**
   - Embeds 2D points (x, y) into a higher-dimensional space.
   - Processes sequences with Transformer layers.
   - Outputs mean (`mu`) and log variance (`logvar`) for latent representation.

2. **Latent Space (VAE Reparameterization)**
   - Uses `mu` and `logvar` to sample latent vectors via the reparameterization trick.

3. **Decoder (TransformerDecoder)**
   - Outputs generated Doom map.

## 📊 Training Details
- **Dataset:** SVG files converted into sequences of 2D points.
- **Loss Function:** VAE loss = MSE (Reconstruction Loss) + KL Divergence.
- **Optimizer:** Adam with a learning rate of `1e-4`.
- **Training Data Size:** At least **10,000 Doom maps** recommended for viable results.
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

## Notes
- If results are poor, increase training data or tune hyperparameters.
- For best results, use at least **10,000+ maps**.
