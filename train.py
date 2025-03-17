import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
LATENT_DIM = 32
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 4
SEQ_LEN = 128  # Max sequence length
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

# Custom Dataset (Placeholder for real data)
class ShapeDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.data = [self.generate_shape() for _ in range(num_samples)]

    def generate_shape(self):
        num_points = np.random.randint(10, SEQ_LEN - 2)
        points = np.random.rand(num_points, 2)  # (x, y) coordinates
        terminator = np.array([[0, 0]])
        end_sequence = np.array([[1, 1]])
        shape = np.vstack([points, terminator, end_sequence])
        padding = np.zeros((SEQ_LEN - shape.shape[0], 2))
        return np.vstack([shape, padding])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(2, EMBED_DIM)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS),
            num_layers=NUM_LAYERS
        )
        self.fc_mu = nn.Linear(EMBED_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(EMBED_DIM, LATENT_DIM)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

# Transformer Decoder
class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, EMBED_DIM)
        self.embedding = nn.Linear(2, EMBED_DIM)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS),
            num_layers=NUM_LAYERS
        )
        self.fc_out = nn.Linear(EMBED_DIM, 2)

    def forward(self, z, seq):
        z = self.fc(z).unsqueeze(0).repeat(SEQ_LEN, 1, 1)
        seq = self.embedding(seq).permute(1, 0, 2)
        out = self.transformer(z, seq)
        return self.fc_out(out.permute(1, 0, 2))

# VAE Model
class ShapeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z, x)
        return recon_x, mu, logvar

# Loss Function
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Training
def train():
    dataset = ShapeDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ShapeVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader.dataset)}")

    torch.save(model.state_dict(), "shape_vae.pth")
    print("Model saved!")

# Generate New Shapes
def generate():
    model = ShapeVAE().to(device)
    model.load_state_dict(torch.load("shape_vae.pth", map_location=device))
    model.eval()

    z = torch.randn(1, LATENT_DIM).to(device)
    start_seq = torch.zeros(1, SEQ_LEN, 2).to(device)  # Ensure batch size is 1
    with torch.no_grad():
        generated_shape = model.decoder(z, start_seq)
    print(generated_shape.cpu().numpy())

if __name__ == "__main__":
    train()
    generate()
