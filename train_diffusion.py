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
import pandas as pd
import datetime

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
spacing     = config['spacing']
filters     = config['filters']
num_epochs  = config['num_epochs']
wd          = config['wd']
lr          = config['lr']
batch_size = config['batch_size']



# --------------------------

iterations = 2101000
# Constants and configurations
on_remote = True  # Flag to switch between remote and local paths
autoencoder_date = '2024-05-24'  # Date of the experiment

autoencoder_model = 'autoencoder-f4-f64'
pth = f'/proj/berzelius-2022-164/users/sm_maran/results/{autoencoder_date}' if on_remote else f'C:/Users/svart/Desktop/MEX/results/{autoencoder_date}'

autoencoder_path = Path(f'{pth}/{autoencoder_model}') if on_remote else Path(f'{pth}/{autoencoder_model}')

# Path to the dataset, changes based on the execution environment

date = '2024-05-26'  # Date of the experiment
result_path = Path(f'/proj/berzelius-2022-164/users/sm_maran/results/{date}/{name}/') if on_remote else Path(f'C:/Users/svart/Desktop/MEX/results/{name}')

# Check if the directory exists, and create it if it doesn't
if not result_path.exists():
    result_path.mkdir(parents=True, exist_ok=True)

# Copy the JSON configuration file to the results directory
config_file_name = Path(config_path).name
shutil.copy(config_path, result_path / "config.json")


# Note that this needs to be precalculated
std_path = f'{autoencoder_path}/WB_stds.txt'
precomputed_std = torch.tensor(np.loadtxt(std_path,delimiter=' ')[:,1], dtype=torch.float32).to(device)
def residual_scaling(x):
    return precomputed_std[x.to(dtype=int)-1]

# ---------------------------

""" 
QG
"""

# QG Dataset

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


""" 
WB
"""

# WB Dataset

offset = 2**7

mean_data = 54112.887 
std_data = 3354.9524
norm_factors = (mean_data, std_data)

spinup = 0
ti = pd.date_range(datetime.datetime(1979,1,1,0), datetime.datetime(2018,12,31,23), freq='1h')
n_train = sum(ti.year <= 2015)
n_val = sum((ti.year >= 2016) & (ti.year <= 2017))
n_samples = len(ti)
sample_counts = (n_samples, n_train, n_val)

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


# Way to load a dataset with lead time following a distribution given by update_k_per_batch
update_k_per_batch = get_uniform_k_dist_fn(kmin=1, kmax=max_lead_time, d=1)

train_time_dataset = QGDataset(lead_time=max_lead_time, dataset_mode='train', **kwargs)
train_batch_sampler = DynamicKBatchSampler(train_time_dataset, batch_size=batch_size, drop_last=True, k_update_callback=update_k_per_batch, shuffle=True)
train_time_loader = DataLoader(train_time_dataset, batch_sampler=train_batch_sampler)

val_time_dataset = QGDataset(lead_time=max_lead_time, dataset_mode='val', **kwargs)
val_batch_sampler = DynamicKBatchSampler(val_time_dataset, batch_size=batch_size, drop_last=True, k_update_callback=update_k_per_batch, shuffle=True)
val_time_loader = DataLoader(val_time_dataset, batch_sampler=val_batch_sampler)

def train():

    """
    Autoencoder
    """
    saved_model = torch.load(autoencoder_path / 'best_model.pth')

    # Read parameters from JSON file
    with open(autoencoder_path / 'config.json', 'r') as json_file:
        parameters = json.load(json_file)

    # Extract the desired parameters
    ae_filters = parameters['filters']
    if 'no_latent_channels' in parameters.keys():
        latent_dim = parameters['no_latent_channels']
    else:
        latent_dim = 1

    # Create an instance of the ConvolutionalAutoencoder class
    autoencoder = DeepAutoencoder(filters= ae_filters, no_latent_channels=latent_dim, no_downsamples=1, start_kernel=3)

    autoencoder.load_state_dict(saved_model)

    autoencoder.to(device)
    autoencoder.eval()
    print("Autoencoder loaded successfully!", flush=True)

    #----------

    """
    Model Loading
    """

    model = EDMPrecond(filters=filters, img_channels=2*latent_dim, img_resolution = 16, time_emb=1, model_type='standard', sigma_data=1, sigma_min=0.02, sigma_max=88)
    loss_fn = GCLoss()

    print("Num params: ", sum(p.numel() for p in model.parameters()), flush=True)
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

        for previous, current, time_label in (train_time_loader):
            current = current.to(device)
            previous = previous.to(device)
            time_label = time_label.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                current_latent = autoencoder.encoder(current)

                previous_latent = autoencoder.encoder(previous)
                target_latent = (current_latent - previous_latent) / residual_scaling(time_label[0])
                
            loss = loss_fn(model, target_latent, previous_latent, time_label/max_lead_time)

            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()        
            warmup_scheduler.step()
    
        avg_train_loss = total_train_loss / len(train_time_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for previous, current, time_label in (val_time_loader):
                current = current.to(device)
                previous = previous.to(device)
                time_label = time_label.to(device)
                
                current_latent = autoencoder.encoder(current)
                previous_latent = autoencoder.encoder(previous)
                target_latent = (current_latent - previous_latent) / residual_scaling(time_label[0])
                
                loss = loss_fn(model, target_latent, previous_latent, time_label/max_lead_time)
                
                total_val_loss += loss.item()
                    
            avg_val_loss = total_val_loss / len(val_time_loader)

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
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}', flush=True)

        torch.save(model.state_dict(), result_path/f'final_model.pth')

train()