# Importing necessary libraries
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import csv

# Constants and configurations
iterations = 2101000  # Total number of iterations for data generation
on_remote = False  # Flag to switch between remote and local paths

# Path to the dataset, changes based on the execution environment
data_path = Path(f'/nobackup/smhid20/users/sm_maran/dpr_data/simulations/QG_samples_SUBS_{iterations}.npy') if on_remote else Path(f'C:/Users/svart/Desktop/MEX/data/QG_samples_SUBS_{iterations}.npy')
k = 1  # Step size for the dataset generation
spinup = 1001  # Number of initial samples to skip
p_train = 0.8  # Proportion of data used for training
mean_data = 0.003394413273781538  # Mean of the dataset, for normalization
std_data = 9.174626350402832  # Standard deviation of the dataset, for normalization

spacing = 50  # Spacing between the samples in the dataset

# Model configuration
filters = 16  # Number of filters in the first convolutional layer of the model
latent_dim = 100  # Dimensionality of the latent space
no_downsamples = 2  # Number of downsampling operations in the model

num_epochs = 500  # Number of epochs for training


class QGSamplesDataset(Dataset):
    """
    Custom Dataset class for loading QG samples.
    Lazily loads data to manage memory usage efficiently.
    """
    def __init__(self, data_path, mode, p_train, k, spinup, spacing, iterations, mean_data, std_data, device, transform=None):
        """
        Initializes the dataset object.
        
        Parameters:
        - data_path (Path): Path to the dataset files.
        - mode (str): Mode of the dataset to be loaded ('train', 'val', 'test').
        - p_train (float): Percentage of data to be used for training.
        - k, spinup, spacing, iterations: Parameters for data generation.
        - mean_data, std_data: Normalization parameters.
        - device (torch.device): Device to load the tensors onto.
        - transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Initialization and dataset configuration
        self.data_path = data_path
        self.data_dtype = 'float32'
        self.device = device
        self.mode = mode
        self.p_train = p_train
        self.k = k
        self.spinup = spinup
        self.spacing = spacing
        self.iterations = iterations
        self.mean_data = mean_data
        self.std_data = std_data
        self.transform = transform

        # Calculate dataset dimensions and generate indices for data access
        self.total_rows, self.total_columns = self._calculate_dimensions()
        self.X_indices, self.Y_indices = self._generate_indices()
        self.mmap = self.create_mmap()  # Memory-map the dataset file for efficient data loading

    
    def create_mmap(self):
        return np.memmap(self.data_path, dtype=self.data_dtype, mode='r', shape=(self.total_rows, self.total_columns))


    def _calculate_dimensions(self):
        total_rows = self.iterations + 1
        total_columns = 4225 
        return total_rows, total_columns


    def _generate_indices(self):
        n_train = int(np.round(self.p_train * (self.total_rows - self.spinup)))  # Number of training samples
        n_val = int(np.round((1 - self.p_train) / 2 * (self.total_rows - self.spinup)))  # Number of validation samples
        # Assuming the remaining samples are used for testing

        if self.mode == 'train':
            start, stop = self.spinup, self.spinup + n_train
        elif self.mode == 'val':
            start, stop = self.spinup + n_train, self.spinup + n_train + n_val
        elif self.mode == 'test':
            start, stop = self.spinup + n_train + n_val, self.total_rows

        fit_x, fit_y = slice(start, stop - self.k), slice(start + self.k, stop)

        # Adjust indices for spacing if necessary
        X_indices = np.arange(fit_x.start, fit_x.stop)[::self.spacing]
        Y_indices = np.arange(fit_y.start, fit_y.stop)[::self.spacing]

        return X_indices, Y_indices


    def __len__(self):
        # Return the length of the dataset
        return self.X_indices.shape[0]

    def __getitem__(self, idx):
        # Calculate the actual data indices for X and Y based on the provided dataset index
        x_index = self.X_indices[idx]
        y_index = self.Y_indices[idx]

        # Access the specific samples directly from the memory-mapped array
        # This operation does not load the entire dataset into memory
        X_sample = self.mmap[x_index, :].astype(self.data_dtype)
        Y_sample = self.mmap[y_index, :].astype(self.data_dtype)

        # Normalize the samples if needed
        X_sample = (X_sample - self.mean_data) / self.std_data
        Y_sample = (Y_sample - self.mean_data) / self.std_data

        # Convert to tensors before returning
        X_sample = torch.tensor(X_sample, dtype=torch.float32).to(self.device)
        Y_sample = torch.tensor(Y_sample, dtype=torch.float32).to(self.device)

        return X_sample, Y_sample

class ConvolutionalAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for encoding and decoding images.
    """
    def __init__(self, filters, latent_dim=100, no_downsamples=2):
        """
        Initializes the model with the specified configuration.
        
        Parameters:
        - filters (int): Number of filters in the first convolutional layer.
        - latent_dim (int): Dimensionality of the latent space.
        - no_downsamples (int): Number of downsampling steps in the encoder.
        """
        super(ConvolutionalAutoencoder, self).__init__()
        self.image_size = 65  # Assuming square images of size 65x65
        self.filters = filters
        self.no_downsamples = no_downsamples
        self.latent_dim = latent_dim


        dim = self.filters

        encoder_layers = [
            nn.Unflatten(1, (1,self.image_size, self.image_size)),
            nn.Conv2d(in_channels=1, out_channels=self.filters, kernel_size=4, padding=1),
            nn.ReLU(True),
        ]

        for _ in range(self.no_downsamples):
            #encoder_layers.append(self._block(dim, dim, kernel_size=3, stride=1))
            encoder_layers.extend(self._block(dim, dim*2, kernel_size=3, stride=1))
            encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            dim *= 2

        enc_img_sz = self.image_size//2**self.no_downsamples
        
        encoder_layers.append(nn.Flatten(start_dim=1))
        encoder_layers.append(nn.Linear(in_features=dim * enc_img_sz * enc_img_sz, out_features=self.latent_dim))
        #encoder_layers.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.Linear(in_features=self.latent_dim, out_features=dim * enc_img_sz * enc_img_sz),
            nn.Unflatten(1, (dim, enc_img_sz, enc_img_sz)),
        ]
        
        for _ in range(self.no_downsamples):
            decoder_layers.extend(self._upblock(dim, dim, kernel_size=4, stride=2))
            decoder_layers.extend(self._block(dim, dim//2, kernel_size=3, stride=1))
            #decoder_layers.append(self._block(dim//2, dim//2, kernel_size=3, stride=1))
            dim //= 2

        decoder_layers.append(nn.ConvTranspose2d(in_channels=self.filters, out_channels=1, kernel_size=4, padding=1))
        decoder_layers.append(nn.Flatten(start_dim=1))
        self.decoder = nn.Sequential(*decoder_layers)
        
        self.apply(self.init_weights)

    def _block(self, in_channels, out_channels, kernel_size, stride):
        return [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
    
    def _upblock(self, in_channels, out_channels, kernel_size, stride):
        return [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
    
    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.encoder(x)
        activations = x  # Store the activations from the encoder
        x = self.decoder(x)
        return x, activations
    
# Setup for training and validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize dataset and dataloader for training and validation
train_dataset = QGSamplesDataset(data_path, 'train', p_train, k, spinup, spacing, iterations, mean_data, std_data, device)
val_dataset = QGSamplesDataset(data_path, 'val', p_train, k, spinup, spacing, iterations, mean_data, std_data, device)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model, loss function, and optimizer
model = ConvolutionalAutoencoder(filters=filters, latent_dim=latent_dim, no_downsamples=no_downsamples).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Additional variables for tracking progress
loss_values = []
val_loss_values = []
best_val_loss = float('inf')
log_file_path = 'training_log.csv'


with open(log_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['Epoch', 'Average Training Loss', 'Validation Loss'])

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_train_loss = 0
    
    for data, _ in train_loader:
        img = data

        optimizer.zero_grad()

        output, activations = model(img)

        reconstruction_loss = criterion(output, img)
        loss = reconstruction_loss 

        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation phase
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    with torch.no_grad():
        for data,_ in val_loader:
            img = data.to(device)

            output, _ = model(img)
            loss = criterion(output, img)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    # Checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')

    loss_values.append([avg_train_loss])
    val_loss_values.append(avg_val_loss)  
    
    # Log to CSV
    with open(log_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, avg_train_loss, avg_val_loss])
    
    #print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
