# pip install torch numpy matplotlib

import torch
import torch.nn as nn
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer-based Autoencoder Model
class ShapeAutoencoder(nn.Module):
    def __init__(self, embed_dim=16, num_heads=2, num_layers=2, latent_dim=8):
        super(ShapeAutoencoder, self).__init__()

        # Encoder: Maps input shapes to a latent space
        self.encoder = nn.Linear(2, embed_dim)

        # Transformer expects (batch, seq_len, embed_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads),
            num_layers=num_layers
        )

        # Decoder: Maps latent space back to shape
        self.decoder = nn.Linear(embed_dim, 2)

    def forward(self, points):
        x = self.encoder(points)  # Encode (batch, seq_len, embed_dim)
        x = self.transformer(x)   # Transformer processes (batch, seq_len, embed_dim)
        x = self.decoder(x)       # Decode back to (batch, seq_len, 2)
        return x


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

# Prepare dataset
shapes = [square(), circle()]
max_len = max(len(shape) for shape in shapes)
padded_shapes = np.array([np.pad(shape, ((0, max_len - len(shape)), (0, 0)), mode='constant') for shape in shapes])
shapes_tensor = torch.tensor(padded_shapes, dtype=torch.float32).to(device)

# Initialize model
model = ShapeAutoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Training loop
epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(shapes_tensor)
    loss = criterion(output, shapes_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "polygon_autoencoder.pth")
print("Model trained and saved successfully!")
