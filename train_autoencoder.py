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
from tqdm import tqdm
import datetime
import pandas as pd

from utils import *
from autoencoder_networks import *
from loss import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
name        = config['name']
model_choice = config['model']
loss_function = config['loss_function']
filters     = config['filters']
wd          = config['wd']
lr          = config['lr']
num_epochs  = config['num_epochs']
spacing     = config['spacing']
l1_penalty = config['l1_penalty']

# --------------------------

iterations = 2101000
# Constants and configurations
on_remote = False  # Flag to switch between remote and local paths

# Path to the dataset, changes based on the execution environment
data_path = Path(f'/nobackup/smhid20/users/sm_maran/dpr_data/simulations/QG_samples_SUBS_{iterations}.npy') if on_remote else Path(f'C:/Users/svart/Desktop/MEX/data/QG_samples_SUBS_{iterations}.npy')

date = '2024-05-23'  # Date of the experiment
result_path = Path(f'/nobackup/smhid20/users/sm_maran/results/{date}/{name}/') if on_remote else Path(f'C:/Users/svart/Desktop/MEX/results/{name}')

# Check if the directory exists, and create it if it doesn't
if not result_path.exists():
    result_path.mkdir(parents=True, exist_ok=True)

# Copy the JSON configuration file to the results directory
config_file_name = Path(config_path).name
shutil.copy(config_path, result_path / "config.json")

# ---------------------------

""" 
QG
"""

# QG Dataset

batch_size = 64 # 256 Largest possible batch size that fits on the GPU w.f32

mean_data = 0.003394413273781538
std_data = 9.174626350402832
norm_factors = (mean_data, std_data)

iterations = 2101000
spinup = 1001
p_train = 0.8

n_samples = iterations+1
n_train = int(np.round(p_train * (n_samples - spinup)))  # Number of training samples
n_val = int(np.round((1 - p_train) / 2 * (n_samples - spinup)))  # Number of validation samples
sample_counts = (n_samples, n_train, n_val)

on_remote = False
fname= f'QG_samples_SUBS_{iterations}.npy'
subd = 'C:/Users/svart/Desktop/MEX/data/'
if on_remote:
    subd = '/nobackup/smhid20/users/sm_maran/dpr_data/simulations'
dataset_path = Path(f'{subd}/{fname}')

grid_dimensions = (65, 65)
max_lead_time = 150

QG_kwargs = {
            'dataset_path':     dataset_path,
            'sample_counts':    sample_counts,
            'grid_dimensions':  grid_dimensions,
            'max_lead_time':    max_lead_time,
            'norm_factors':     norm_factors,
            'device':           device,
            'spinup':           spinup,
            'spacing':          spacing,
            'dtype':            'float32'
            }
# WB
# ---------------------------

""" 
WB
"""

# WB Dataset

batch_size = 32 # 256 Largest possible batch size that fits on the GPU w.f32
offset = 2**7

mean_data = 0.003394413273781538
std_data = 9.174626350402832
norm_factors = (mean_data, std_data)

spinup = 0
ti = pd.date_range(datetime.datetime(1979,1,1,0), datetime.datetime(2018,12,31,23), freq='1h')
n_train = sum(ti.year <= 2015)
n_val = sum((ti.year >= 2016) & (ti.year <= 2017))
n_samples = len(ti)
sample_counts = (n_samples, n_train, n_val)

on_remote = False
fname= 'geopotential_500hPa_1979-2018_5.625deg.npy'
subd = 'C:/Users/svart/Desktop/MEX/data/'
if on_remote:
    #subd = '/nobackup/smhid20/users/sm_maran/dpr_data/simulations'
    subd = '/proj/berzelius-2022-164/users/sm_maran/data/wb'
dataset_path = Path(f'{subd}/{fname}')

grid_dimensions = (32, 64)
max_lead_time = 240

fnm_ll = f'{subd}/latlon_500hPa_1979-2018_5.625deg.npz'
buf = np.load(fnm_ll)
lat, lon = buf['arr_0'], buf['arr_1']

wmse = AreaWeightedMSELoss(lat, lon, device).loss_fn


WB_kwargs = {
            'dataset_path':     dataset_path,
            'sample_counts':    sample_counts,
            'grid_dimensions':  grid_dimensions,
            'max_lead_time':    max_lead_time,
            'norm_factors':     norm_factors,
            'device':           device,
            'spinup':           spinup,
            'spacing':          spacing,
            'dtype':            'float32',
            'offset':           offset
            }

kwargs = WB_kwargs

# ---------------------------
# Way to load a dataset with a specific lead time
lead_time = 1
train_dataset = QGDataset(lead_time=lead_time,dataset_mode='train', **kwargs)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Way to load a dataset with a specific lead time
val_dataset = QGDataset(lead_time=lead_time, dataset_mode='val', **kwargs)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

def train():


    """
    Model Loading
    """
    if model_choice == "autoencoder":
        model = Autoencoder(filters= filters, no_latent_channels=1, no_downsamples=2, start_kernel=3)
    elif model_choice== "deep":
        model = DeepAutoencoder(filters= filters, no_latent_channels=1, no_downsamples=2, start_kernel=3)

    if loss_function == "MSE":
        loss_fn = nn.MSELoss()
    elif loss_function == "L1":
        loss_fn = nn.L1Loss()
    elif loss_function == "WMSE":
        loss_fn = wmse
    else:
        raise Exception("Loss function not found")

    print("Num params: ", sum(p.numel() for p in model.parameters()))
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=1000)

    loss_values = []
    val_loss_values = []
    best_val_loss = float('inf')

    # Setup for logging
    log_file_path = result_path / f'training_log.csv'
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Average Training Loss', 'Validation Loss'])
    

    """
    Training starts here
    """
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0

        for previous, _, _ in tqdm(train_loader):
            previous = previous.to(device)
            
            optimizer.zero_grad()
            
            reconstruction, latent = model(previous)

            reconstruction_loss = loss_fn(reconstruction, previous)
            latent_loss = l1_penalty * torch.mean(torch.norm(latent, 1, dim=1))

            loss = reconstruction_loss + latent_loss
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()        
            warmup_scheduler.step()
    
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for previous, _, _ in tqdm(val_loader):
                previous = previous.to(device)
                        
                reconstruction, latent = model(previous)

                reconstruction_loss = loss_fn(reconstruction, previous)
                latent_loss = l1_penalty * torch.mean(torch.norm(latent, 1, dim=1))

                loss = reconstruction_loss + latent_loss
                
                total_val_loss += loss.item()
                    
            avg_val_loss = total_val_loss / len(val_loader)

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), result_path/f'best_model.pth')
        
        scheduler.step()
        
        # Log to CSV    
        loss_values.append([avg_train_loss])
        val_loss_values.append(avg_val_loss)  # Assuming val_loss_values list exists
        
        # Log to CSV
        with open(log_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss])
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    torch.save(model.state_dict(), result_path/f'final_model.pth')

train()