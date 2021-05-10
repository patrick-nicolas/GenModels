from tqdm import tqdm

def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, dataloader, dataset, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)  # reconstruction[64, 1, 32, 32]  data [64, 1, 32, 32]
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter
    return train_loss

def validate(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset) / dataloader.batch_size)):
            counter += 1
            data = data[0]
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(dataset) / dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

kernel_size = 4 # (4, 4) kernel
init_channels = 8 # initial number of filters
image_channels = 1 # MNIST images are grayscale
latent_dim = 16 # latent dimension for sampling
conv_hidden_dim = 64
linear_hidden_dim = 72

class ConvVAE(nn.Module):
    def __init__(self):
        super(ConvVAE, self).__init__()
        # encoder_model
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size,stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels * 2, kernel_size=kernel_size,stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels * 2, out_channels=init_channels * 4, kernel_size=kernel_size,stride=2, padding=1
        )

        self.enc4 = nn.Conv2d(
            in_channels=init_channels * 4, out_channels=conv_hidden_dim, kernel_size=kernel_size,stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(conv_hidden_dim, linear_hidden_dim)
        self.fc_mu = nn.Linear(linear_hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(linear_hidden_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, conv_hidden_dim)
        # decoder_model
        self.dec1 = nn.ConvTranspose2d(
            in_channels=conv_hidden_dim, out_channels=init_channels * 8, kernel_size=kernel_size,stride=1, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels * 8, out_channels=init_channels * 4, kernel_size=kernel_size,stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels * 4, out_channels=init_channels * 2, kernel_size=kernel_size,stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels * 2, out_channels=image_channels, kernel_size=kernel_size,stride=2, padding=1
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # standard deviation
        eps = torch.randn_like(std)     # `randn_like` as we need the same size
        sample = mu + (eps * std)       # sampling
        return sample

    def forward(self, x):
        # encoding  x                    [64, 1, 32, 32]
        x = F.relu(self.enc1(x))       # [64, 8, 16, 16]
        x = F.relu(self.enc2(x))       # [64, 16, 8, 8]
        x = F.relu(self.enc3(x))       # [64, 32, 4, 4]
        x = F.relu(self.enc4(x))       # [64, 64, 1, 1]
        batch, _, _, _ = x.shape                             # Batch => 64
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)   # [64, 64]
        hidden = self.fc1(x)                                 # [64, 128]
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)                              # [64, 16]
        log_var = self.fc_log_var(hidden)
        # get the latent vector through re-parameterization
        z = self.reparameterize(mu, log_var)                 # [64, 16]
        z = self.fc2(z)                                      # [64, 64]
        z = z.view(-1, conv_hidden_dim, 1, 1)                             # [64, 64, 1, 1]
        # decoding
        x = F.relu(self.dec1(z))                             # [64, 64, 4, 4]
        x = F.relu(self.dec2(x))                             # [64, 32, 8, 8]
        x = F.relu(self.dec3(x))                             # [64, 16, 16, 16]
        reconstruction = torch.sigmoid(self.dec4(x))         # [64, 1, 32, 32]
        # reconstruction = torch.sigmoid(self.dec3(x))
        return reconstruction, mu, log_var


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initialize the model
model = ConvVAE()
# set the learning parameters
lr = 0.001
epochs = 2
batch_size = 30
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')
# a list to save all the reconstructed images in PyTorch grid format
grid_images = []

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
# training set and train data loader
trainset = torchvision.datasets.MNIST(
    root='../cnn/test/MNIST', train=True, download=True, transform=transform
)
trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)
# validation set and validation data loader
testset = torchvision.datasets.MNIST(
    root='../cnn/test/MNIST', train=False, download=True, transform=transform
)
testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=False
)


train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, trainloader, trainset, device, optimizer, criterion
    )
    valid_epoch_loss, recon_images = validate(
        model, testloader, testset, device, criterion
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop

    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")