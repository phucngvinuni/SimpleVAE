import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from model import VAE
from tqdm import tqdm
from torchvision.utils import save_image
from torch import nn, optim
import os
import matplotlib.pyplot as plt

# Config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
INPUT_DIM = 784
H_DIM = 1024
Z_DIM = 100
NUM_EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 5e-4

# Dataset
dataset = datasets.MNIST(root='data/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
model = VAE(inputdim=INPUT_DIM, hiddendim=H_DIM, latentdim=Z_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.BCELoss(reduction='sum')
beta = 1  # KL divergence weight
# Create output directory
os.makedirs('outputs', exist_ok=True)
train_losses = []
# Training function
def train_fn():

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for i, (x, _) in loop:
            # Forward
            x = x.to(DEVICE).view(x.shape[0], INPUT_DIM)
            x_reconstructed, mu, sigma = model(x)
            x_reconstructed = torch.sigmoid(x_reconstructed)

            # Compute loss
            reconstruction_loss = loss_fn(x_reconstructed, x) / x.size(0)
            kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            loss = reconstruction_loss + beta * kl_div  # Add beta for KL divergence
            epoch_loss += loss.item()
            train_losses.append(epoch_loss/len(train_loader))
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

# Inference function
def inference_fn(digit, num_examples=1):
    images = [x for x, y in dataset if y == digit][:num_examples]
    for idx, img in enumerate(images):
        with torch.no_grad():
            mu, sigma = model.encode(img.view(-1, INPUT_DIM).to(DEVICE))
            epsilon = torch.randn_like(sigma)
            z = mu + sigma * epsilon
            out = model.decode(z)
            save_image(out.view(28, 28), f'outputs/output_{digit}_{idx}.png')

# Main
if __name__ == '__main__':
    print(f"Training on: {DEVICE}")
    train_fn()
    torch.save(model.state_dict(), 'model.pth')
    print('Model saved')
    for idx in range(10):
        inference_fn(idx, num_examples=1)

    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
