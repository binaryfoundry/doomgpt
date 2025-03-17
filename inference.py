import torch
import numpy as np
import matplotlib.pyplot as plt
from train import ShapeVAE  # Assuming model is saved in 'model.py'

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LATENT_DIM = 32
SEQ_LEN = 128

# Load the trained model
def load_model(model_path="shape_vae.pth"):
    model = ShapeVAE().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Generate new shapes
def generate_shapes(model, num_samples=5):
    generated_shapes = []
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, LATENT_DIM).to(device)  # Sample from latent space
            start_seq = torch.zeros(1, SEQ_LEN, 2).to(device)  # Empty sequence
            generated_shape = model.decoder(z, start_seq).cpu().numpy()
            generated_shapes.append(generated_shape.squeeze(0))  # Remove batch dim
    return generated_shapes

# Plot generated shapes
def plot_shapes(shapes):
    plt.figure(figsize=(10, 5))
    for i, shape in enumerate(shapes):
        plt.subplot(1, len(shapes), i + 1)
        plt.plot(shape[:, 0], shape[:, 1], marker='o', linestyle='-')
        plt.title(f"Shape {i+1}")
        plt.axis("equal")
    plt.show()

if __name__ == "__main__":
    model = load_model()
    shapes = generate_shapes(model, num_samples=5)
    plot_shapes(shapes)
