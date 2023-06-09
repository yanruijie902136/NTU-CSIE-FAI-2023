import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

"""
Implementation of Autoencoder
"""
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Linear(encoding_dim, encoding_dim//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim//2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )

    def forward(self, x):
        #TODO: 5%
        # Hint: a forward pass includes one pass of encoder and decoder
        return self.decoder(self.encoder(x))
        # raise NotImplementedError

    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 5%
        # Hint: a regular pytorch training includes:
        # 1. define optimizer
        # 2. define loss function
        # 3. define number of epochs
        # 4. define batch size
        # 5. define data loader
        # 6. define training loop
        # 7. record loss history 
        # Note that you can use `self(X)` to make forward pass.

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        loss_function = nn.MSELoss()
        data_loader = DataLoader(
                dataset=TensorDataset(torch.tensor(X, dtype=torch.float)),
                batch_size=batch_size,
                shuffle=False
        )

        # loss_history = []
        for epoch in range(epochs):
            # epoch_loss = 0
            for batch in data_loader:
                batch_tensor = torch.cat(batch)
                optimizer.zero_grad()
                loss = loss_function(batch_tensor, self.forward(batch_tensor))
                loss.backward()
                optimizer.step()
                # epoch_loss += loss

            # epoch_loss /= len(data_loader.dataset)
            # loss_history.append(epoch_loss.item())

        # plt.plot(loss_history)
        # plt.savefig("curve_ae")
        # plt.clf()

        # raise NotImplementedError

    def transform(self, X):
        #TODO: 2%
        #Hint: Use the encoder to transofrm X
        return self.encoder(torch.tensor(X, dtype=torch.float)).detach().numpy()
        # raise NotImplementedError
    
    def reconstruct(self, X):
        #TODO: 2%
        #Hint: Use the decoder to reconstruct transformed X
        return self.decoder(torch.tensor(self.transform(X), dtype=torch.float)).detach().numpy()
        # raise NotImplementedError


"""
Implementation of DenoisingAutoencoder
"""
class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim,encoding_dim)
        self.noise_factor = noise_factor

    def add_noise(self, x):
        #TODO: 3%
        #Hint: Generate Gaussian noise with noise_factor
        mean = torch.zeros(x.size())
        std = torch.zeros(x.size()) + self.noise_factor
        return x + torch.normal(mean=mean, std=std)
        # raise NotImplementedError

    def fit(self, X, epochs=10, batch_size=32):
        #TODO: 4%
        #Hint: Follow the same procedure above but remember to add_noise before training.

        optimizer = optim.Adam(self.parameters(), lr=0.001)
        # optimizer = optim.Adagrad(self.parameters(), lr=0.001)
        # optimizer = optim.SGD(self.parameters(), lr=0.001)
        # optimizer = optim.RMSprop(self.parameters(), lr=0.001)
        loss_function = nn.MSELoss()
        data_loader = DataLoader(
                dataset=TensorDataset(torch.tensor(X, dtype=torch.float)),
                batch_size=batch_size,
                shuffle=False
        )

        # loss_history = []
        for epoch in range(epochs):
            # epoch_loss = 0
            for batch in data_loader:
                batch_tensor = torch.cat([self.add_noise(x) for x in batch])
                optimizer.zero_grad()
                loss = loss_function(batch_tensor, self.forward(batch_tensor))
                loss.backward()
                optimizer.step()
                # epoch_loss += loss

            # epoch_loss /= len(data_loader.dataset)
            # loss_history.append(epoch_loss.item())

        # plt.plot(loss_history)
        # plt.savefig("curve_deno_ae_rmsprop")
        # plt.clf()

        # raise NotImplementedError
