import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Variational Transformer Autoencoder
class VariationalShapeAutoencoder(nn.Module):
    def __init__(self, embed_dim=16, num_heads=2, num_layers=2):
        super(VariationalShapeAutoencoder, self).__init__()

        # Encoder: Maps input shapes to latent mean and variance
        self.encoder = nn.Linear(2, embed_dim)
        self.mu_layer = nn.Linear(embed_dim, embed_dim)  # Mean of latent distribution
        self.logvar_layer = nn.Linear(embed_dim, embed_dim)  # Log-variance for reparameterization

        # Transformer processing latent space
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads), 
            num_layers=num_layers
        )

        # Decoder: Maps latent space back to shape
        self.decoder = nn.Linear(embed_dim, 2)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, sigma^2)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Random normal noise
        return mu + eps * std

    def encode(self, points):
        x = self.encoder(points)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar  # Return latent sample and distribution parameters

    def decode(self, z):
        z = self.transformer(z)
        return self.decoder(z)

    def forward(self, points):
        z, mu, logvar = self.encode(points)
        return self.decode(z), mu, logvar  # Output reconstruction and latent parameters

# Load trained model
model = VariationalShapeAutoencoder().to(device)
model.load_state_dict(torch.load("polygon_vae.pth"))
model.eval()
print("Variational Autoencoder model loaded successfully!")

# Define shapes
def square():
    return np.array([
        [1, 1], [0.333, 1], [-0.333, 1], [-1, 1],
        [-1, 0.333], [-1, -0.333], [-1, -1], [-0.333, -1],
        [0.333, -1], [1, -1], [1, -0.333], [1, 0.333]
    ])

def circle():
    rot_offset = np.pi / 4
    angles = np.linspace(rot_offset, (2 * np.pi) + rot_offset, 12, endpoint=False)
    x, y = np.cos(angles), np.sin(angles)
    return np.column_stack([x, y])

# Encode shape into latent space
def encode_shape(shape):
    shape_tensor = torch.tensor(shape, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        latent, _, _ = model.encode(shape_tensor)
    return latent

# Decode from latent space
def decode_shape(latent):
    with torch.no_grad():
        reconstructed = model.decode(latent)
    return reconstructed.cpu().numpy().squeeze()

# SLERP function for smooth interpolation
def slerp(val, low, high):
    """Spherical Linear Interpolation (SLERP) for smooth latent space transitions."""
    dot = np.sum(low * high, axis=-1) / (np.linalg.norm(low, axis=-1) * np.linalg.norm(high, axis=-1))
    dot = np.clip(dot, -1.0, 1.0)  # Clip for numerical stability
    theta = np.arccos(dot) * val
    sin_theta = np.sin(theta)
    sin_total_theta = np.sin(np.arccos(dot))

    s0 = np.sin((1 - val) * np.arccos(dot)) / (sin_total_theta + 1e-8)
    s1 = sin_theta / (sin_total_theta + 1e-8)
    return s0[:, None] * low + s1[:, None] * high

# Shapes for interpolation
shapes = [square(), circle()]
latent_shapes = [encode_shape(shape).cpu().numpy().squeeze() for shape in shapes]

# Number of interpolation steps
num_steps = 6
fig, axes = plt.subplots(1, num_steps + 2, figsize=(12, 3))

# Plot original shapes
axes[0].plot(shapes[0][:, 0], shapes[0][:, 1], 'bo-', label="Square")
axes[0].set_title("Square")
axes[-1].plot(shapes[-1][:, 0], shapes[-1][:, 1], 'ro-', label="Circle")
axes[-1].set_title("Circle")

# Generate interpolated shapes using SLERP
for i in range(1, num_steps + 1):
    alpha = i / (num_steps + 1)
    interpolated_latent = slerp(alpha, latent_shapes[0], latent_shapes[1])
    interpolated_latent_tensor = torch.tensor(interpolated_latent, dtype=torch.float32).unsqueeze(0).to(device)
    reconstructed_shape = decode_shape(interpolated_latent_tensor)

    axes[i].plot(reconstructed_shape[:, 0], reconstructed_shape[:, 1], 'go-', label=f"α={alpha:.2f}")
    axes[i].set_title(f"α={alpha:.2f}")

# Formatting plots
for ax in axes:
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

plt.show()
