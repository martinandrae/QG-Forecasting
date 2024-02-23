# Importing necessary libraries
# Importing necessary libraries for the analysis.
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
import math
import json  # Import json library
import argparse
import shutil

# Setup argument parser to accept a JSON config file path
parser = argparse.ArgumentParser(description='Run model with configuration from JSON file.')
parser.add_argument('config_path', type=str, help='Path to JSON configuration file.')
args = parser.parse_args()

# Load configuration from a JSON file
def load_config(json_file):
    with open(json_file, 'r') as file:
        config = json.load(file)
    return config

config_path = args.config_path
config = load_config(config_path)

# Constants and configurations loaded from JSON
iterations = config['iterations']
name = config['name']
model_name = config['model']
spacing = config['spacing']
filters = config['filters']
num_epochs = config['num_epochs']
no_downsamples = config['no_downsamples']
k = config['k']

# Ensure the JSON file paths and parameters match your requirements.

# Constants and configurations
on_remote = True  # Flag to switch between remote and local paths
date = '2024-02-21'  # Date of the experiment

autoencoder_model = 'ae-2ds-32f-1l-150e-L1-0wd-0.00001l1' #'ae-3ds-16f-2l-150e-L1-0wd-0.00001l1'# 
autoencoder_path = Path(f'/nobackup/smhid20/users/sm_maran/results/{date}/{autoencoder_model}/') if on_remote else Path(f'C:/Users/svart/Desktop/MEX/results/{date}/{autoencoder_model}/')

# Path to the dataset, changes based on the execution environment
data_path = Path(f'/nobackup/smhid20/users/sm_maran/dpr_data/simulations/QG_samples_SUBS_{iterations}.npy') if on_remote else Path(f'C:/Users/svart/Desktop/MEX/data/QG_samples_SUBS_{iterations}.npy')
result_path = Path(f'/nobackup/smhid20/users/sm_maran/results/{date}/{name}/') if on_remote else Path(f'C:/Users/svart/Desktop/MEX/results/{name}')

# Check if the directory exists, and create it if it doesn't
if not result_path.exists():
    result_path.mkdir(parents=True, exist_ok=True)

# Copy the JSON configuration file to the results directory
config_file_name = Path(config_path).name
shutil.copy(config_path, result_path / "config.json")

spinup = 1001  # Number of initial samples to skip
p_train = 0.8  # Proportion of data used for training
mean_data = 0.003394413273781538  # Mean of the dataset, for normalization
std_data = 9.174626350402832  # Standard deviation of the dataset, for normalization

mean_data_latent = -0.661827802658081
std_data_latent = 5.319980144500732
std_residual_latent = 5.724194526672363

from utils import *
from autoencoder import Autoencoder
from diffusion_networks import *

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = QGSamplesDataset(data_path, 'train', p_train, k, spinup, spacing, iterations, mean_data, std_data, device)
    val_dataset = QGSamplesDataset(data_path, 'val', p_train, k, spinup, spacing, iterations, mean_data, std_data, device)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Both Generation and Forecasting
    import json
    # Load the saved model
    saved_model = torch.load(autoencoder_path / 'best_model.pth')

    # Read parameters from JSON file
    with open(autoencoder_path / 'config.json', 'r') as json_file:
        parameters = json.load(json_file)

    # Extract the desired parameters
    ae_filters = parameters['filters']
    latent_dim = parameters['latent_dim']
    ae_no_downsamples = parameters['no_downsamples']

    # Create an instance of the ConvolutionalAutoencoder class
    autoencoder = Autoencoder(filters= ae_filters, no_latent_channels=latent_dim, ae_no_downsamples=no_downsamples)

    # Load the state_dict of the saved model into the conv_autoencoder
    autoencoder.load_state_dict(saved_model)

    autoencoder.to(device)
    autoencoder.eval()

    print("Autoencoder loaded successfully!")

    if model_name == "generate":
        forecasting = False
        model = GCPrecond(sigma_data=1, filters=filters, no_downsamples=no_downsamples, img_channels=1, img_resolution = 16)
    
    elif model_name == "forecast":
        forecasting = True
        model = GCPrecond(sigma_data=1, filters=filters, no_downsamples=no_downsamples, img_channels=2, img_resolution = 16)
    
    print("Num params: ", sum(p.numel() for p in model.parameters()))

    model.to(device)

    optimizer = optim.Adam(model.parameters())
    loss_fn = GCLoss()

    loss_values = []
    val_loss_values = []
    best_val_loss = float('inf')

    # Setup for logging
    log_file_path = result_path / f'training_log.csv'
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Epoch', 'Average Training Loss', 'Validation Loss'])

    import time
    # Training loop

    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()  # Set model to training mode
        total_train_loss = 0

        for previous, current in train_loader:        
            optimizer.zero_grad()

            with torch.no_grad():
                current_latent = autoencoder.encoder(current)
                previous_latent = None

                if forecasting:
                    previous_latent = autoencoder.encoder(previous)
                    target_latent = (current_latent - previous_latent) / std_residual_latent
                else:
                    target_latent = (current_latent - mean_data_latent) / std_data_latent
                
            loss = loss_fn(model, target_latent, previous_latent)

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        train_time = time.time() - start_time
        start_time = time.time()

        # Validation phase
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for previous, current in val_loader:
                current_latent = autoencoder.encoder(current)
                previous_latent = None

                if forecasting:
                    previous_latent = autoencoder.encoder(previous)
                    target_latent = (current_latent - previous_latent) / std_residual_latent
                else:
                    target_latent = (current_latent - mean_data_latent) / std_data_latent
                
                loss = loss_fn(model, target_latent, previous_latent)
                
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), result_path/f'best_model.pth')

        val_time = time.time() - start_time
        start_time = time.time()
        # scheduler.step()

        sample_time = time.time() - start_time
        
        # Log to CSV    
        loss_values.append([avg_train_loss])
        val_loss_values.append(avg_val_loss)  # Assuming val_loss_values list exists
        
        # Log to CSV
        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss])
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        print(f'Training time: {train_time:.5f}s, Validation time: {val_time:.5f}s, Sample time: {sample_time:.5f}s')

    torch.save(model.state_dict(), result_path/f'final_model.pth')

train()