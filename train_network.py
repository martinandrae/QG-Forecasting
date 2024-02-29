# Importing necessary libraries
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import csv
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
latent_dim = config['latent_dim']
no_downsamples = config['no_downsamples']
num_epochs = config['num_epochs']
loss_function = config['loss_function']
weight_decay = config['weight_decay']
l1_penalty = config['l1_penalty']

# Your existing class definitions and the rest of the script remain unchanged

# Note: Remember to update your JSON file structure to match the variables you're loading. For example:
"""
{
  "iterations": 2101000,
  "model": "fully",
  "name": "fully-4ds-8f-100l-150e-MSE-0wd-0.00001l1",
  "spacing": 1,
  "filters": 8,
  "latent_dim": 100,
  "no_downsamples": 4,
  "num_epochs": 150,
  "loss_function": "MSE",
  "weight_decay": 0,
  "l1_penalty": 0.00001
}
"""

# Ensure the JSON file paths and parameters match your requirements.

# Constants and configurations
on_remote = True  # Flag to switch between remote and local paths
date = '2024-02-21'  # Date of the experiment

# Path to the dataset, changes based on the execution environment
data_path = Path(f'/nobackup/smhid20/users/sm_maran/dpr_data/simulations/QG_samples_SUBS_{iterations}.npy') if on_remote else Path(f'C:/Users/svart/Desktop/MEX/data/QG_samples_SUBS_{iterations}.npy')
result_path = Path(f'/nobackup/smhid20/users/sm_maran/results/{date}/{name}/') if on_remote else Path(f'C:/Users/svart/Desktop/MEX/results/{name}')

# Check if the directory exists, and create it if it doesn't
if not result_path.exists():
    result_path.mkdir(parents=True, exist_ok=True)

# Copy the JSON configuration file to the results directory
config_file_name = Path(config_path).name
shutil.copy(config_path, result_path / "config.json")

k = 1  # Step size for the dataset generation
spinup = 1001  # Number of initial samples to skip
p_train = 0.8  # Proportion of data used for training
mean_data = 0.003394413273781538  # Mean of the dataset, for normalization
std_data = 9.174626350402832  # Standard deviation of the dataset, for normalization

from utils import *
from autoencoder import Autoencoder

def train():
    # Setup for training and validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize dataset and dataloader for training and validation
    train_dataset = QGSamplesDataset(data_path, 'train', p_train, k, spinup, spacing, iterations, mean_data, std_data, device)
    val_dataset = QGSamplesDataset(data_path, 'val', p_train, k, spinup, spacing, iterations, mean_data, std_data, device)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    if model_name == "autoencoder":
        model = Autoencoder(filters=filters, no_latent_channels=latent_dim, no_downsamples=no_downsamples)
    else:
        raise Exception("Model not found")

    if loss_function == "MSE":
        criterion = nn.MSELoss()
    elif loss_function == "L1":
        criterion = nn.L1Loss()
    else:
        raise Exception("Loss function not found")

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)

    # Additional variables for tracking progress
    loss_values = []
    val_loss_values = []
    best_val_loss = float('inf')
    log_file_path = result_path / f'training_log.csv'

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

            loss = criterion(output, img)

            if l1_penalty > 0:
                l1_loss = torch.mean(torch.norm(activations, 1, dim=1))
                loss += l1_penalty * l1_loss
            
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for data,_ in val_loader:
                img = data

                output, _ = model(img)
                loss = criterion(output, img)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), result_path/f'best_model.pth')

        loss_values.append([avg_train_loss])
        val_loss_values.append(avg_val_loss)  
        
        # Log to CSV
        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss])
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
    
    
    torch.save(model.state_dict(), result_path/f'final_model.pth')

train()