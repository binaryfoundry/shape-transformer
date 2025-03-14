# pip install torch numpy matplotlib

import torch
import torch.nn as nn
import numpy as np

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

# Define Shapes
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

# Augmentation: Random Rotation and Flip
def augment_shape(shape):
    # Random rotation
    angle = np.random.uniform(0, 2 * np.pi)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_shape = shape @ rotation_matrix.T

    # Random flip
    if np.random.rand() > 0.5:
        rotated_shape[:, 0] *= -1
    if np.random.rand() > 0.5:
        rotated_shape[:, 1] *= -1

    return rotated_shape

# Prepare dataset
shapes = [square(), circle()]
max_len = max(len(shape) for shape in shapes)

# Apply augmentation
augmented_shapes = [augment_shape(shape) for shape in shapes]

# Pad shapes to ensure uniform size
padded_shapes = np.array([np.pad(shape, ((0, max_len - len(shape)), (0, 0)), mode='constant') for shape in augmented_shapes])
shapes_tensor = torch.tensor(padded_shapes, dtype=torch.float32).to(device)

# Initialize model
model = VariationalShapeAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

def kl_divergence(mu, logvar):
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# Training loop
epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()

    # Apply augmentation during training
    augmented_shapes = [augment_shape(shape) for shape in shapes]
    padded_shapes = np.array([np.pad(shape, ((0, max_len - len(shape)), (0, 0)), mode='constant') for shape in augmented_shapes])
    shapes_tensor = torch.tensor(padded_shapes, dtype=torch.float32).to(device)

    output, mu, logvar = model(shapes_tensor)
    recon_loss = criterion(output, shapes_tensor)
    kl_loss = kl_divergence(mu, logvar)
    loss = recon_loss + 0.001 * kl_loss  # Balance reconstruction and regularization

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()} (Recon: {recon_loss.item()}, KL: {kl_loss.item()})")

# Save model
torch.save(model.state_dict(), "polygon_vae.pth")
print("Variational Autoencoder trained and saved successfully!")
