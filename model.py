import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, inputdim=784, hiddendim=400, latentdim=20):
        super(VAE, self).__init__()
        # Encoder
        self.img_2hid = nn.Linear(inputdim, hiddendim)
        self.hidden_2mu = nn.Linear(hiddendim, latentdim)
        self.hidden_2logvar = nn.Linear(hiddendim, latentdim)

        # Decoder
        self.z_2hid = nn.Linear(latentdim, hiddendim)
        self.hid_2img = nn.Linear(hiddendim, inputdim)

        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.relu(self.img_2hid(x))
        mu = self.hidden_2mu(h)
        logvar = self.hidden_2logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z_reparametrized = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z_reparametrized)
        return x_reconstructed, mu, logvar

if __name__ == '__main__':
    vae = VAE(inputdim=784, hiddendim=400, latentdim=20)
    x = torch.randn(4, 28 * 28)
    x_reconstructed, mu, logvar = vae(x)
    print(x_reconstructed.shape)  # Expected: torch.Size([4, 784])
    print(mu.shape)              # Expected: torch.Size([4, 20])
    print(logvar.shape)          # Expected: torch.Size([4, 20])

    # Check output range
    assert x_reconstructed.min() >= 0 and x_reconstructed.max() <= 1, "Output values should be in [0, 1]"
