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
from loss import *
from sampler import *

# -----------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------

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
max_lead_time = config['max_lead_time']
label_dropout = config['label_dropout']
initial_times = config['initial_times']
dt = config['dt']
model_choice = config['model']

# -----------------------------------------------------

# Path to the dataset, changes based on the execution environment

date = '2024-09-09'  # Date of the experiment
result_path = Path(f'/proj/berzelius-2022-164/users/sm_maran/results/{date}/{name}/')
subd = '/proj/berzelius-2022-164/users/sm_maran/data/wb'


# Check if the directory exists, and create it if it doesn't
if not result_path.exists():
    result_path.mkdir(parents=True, exist_ok=True)

# Copy the JSON configuration file to the results directory
config_file_name = Path(config_path).name
shutil.copy(config_path, result_path / "config.json")

# -----------------------------------------------------

stds_directory = save_directory = f"{subd}/residual_stds"

var_names = ['z500', 't850', 't2m', 'u10', 'v10']

precomputed_std = []

for var_name in var_names:
    file_path = f'{stds_directory}/WB_{var_name}.txt'
    std_values = torch.tensor(np.loadtxt(file_path, delimiter=' ')[:, 1], dtype=torch.float32).to(device)
    precomputed_std.append(std_values)

precomputed_std = torch.stack([res_std for res_std in precomputed_std], axis=1)

precomputed_std = precomputed_std[:max_lead_time]

def residual_scaling(x):
    if x.ndim == 0:
        x = x.unsqueeze(0)  
    indices = x.to(dtype=int) - 1
    
    return precomputed_std[indices].view(x.shape[0], vars, 1, 1)

# -----------------------------------------------------

fname= 'z500_t850_t2m_u10_v10_1979-2018_5.625deg.npy'

static_data_path = f'{subd}/orog_lsm_1979-2018_5.625deg.npy'
static_vars = 2

# Load the normalization factors from the JSON file
json_file = f'{subd}/norm_factors.json'
with open(json_file, 'r') as f:
    statistics = json.load(f)

mean_data = torch.tensor([stats["mean"] for stats in statistics.values()])
std_data = torch.tensor([stats["std"] for stats in statistics.values()])

norm_factors = np.stack([mean_data, std_data], axis=0)

vars = len(mean_data)

# -----------------------------------------------------

lead_time_range = [dt, max_lead_time, dt]

random_lead_time = 1 # Yes for random lead time

spacing = spacing
spinup = 0
ti = pd.date_range(datetime.datetime(1979,1,1,0), datetime.datetime(2018,12,31,23), freq='1h')
ti = pd.date_range(datetime.datetime(1979,1,1,0), datetime.datetime(2018,12,31,23), freq='1h')
n_train = sum(ti.year <= 2015)
n_val = sum((ti.year >= 2016) & (ti.year <= 2017))
n_samples = len(ti)
sample_counts = (n_samples, n_train, n_val)

subd = '/proj/berzelius-2022-164/users/sm_maran/data/wb'
dataset_path = Path(f'{subd}/{fname}')

grid_dimensions = (vars, 32, 64)

offset = 2**7

WB_kwargs = {
            'dataset_path':     dataset_path,
            'sample_counts':    sample_counts,
            'dimensions':  grid_dimensions,
            'max_lead_time':    max_lead_time,
            'norm_factors':     norm_factors,
            'device':           device,
            'spinup':           spinup,
            'spacing':          spacing,
            'dtype':            'float32',
            'offset':           offset,
            'initial_times':    initial_times,
            'lead_time_range':  lead_time_range,
            'static_data_path': static_data_path,
            'random_lead_time': random_lead_time,
            }

kwargs = WB_kwargs

# -----------------------------------------------------

mean_data = mean_data.to(device)
std_data = std_data.to(device)
def renormalize(x, mean_ar=mean_data, std_ar=std_data):
    x = x * std_ar[None, :, None, None] + mean_ar[None, :, None, None]
    return x

fnm_ll = f'{subd}/latlon_500hPa_1979-2018_5.625deg.npz'
buf = np.load(fnm_ll)
lat, lon = buf['arr_0'], buf['arr_1']
# -----------------------------------------------------



# Way to load a dataset with lead time following a distribution given by update_k_per_batch
update_k_per_batch = get_uniform_k_dist_fn(kmin=dt, kmax=max_lead_time, d=dt)

train_time_dataset = QGDataset(lead_time=max_lead_time, dataset_mode='train', **kwargs)
train_batch_sampler = DynamicKBatchSampler(train_time_dataset, batch_size=batch_size, drop_last=True, k_update_callback=update_k_per_batch, shuffle=True)
train_time_loader = DataLoader(train_time_dataset, batch_sampler=train_batch_sampler)

val_time_dataset = QGDataset(lead_time=max_lead_time, dataset_mode='val', **kwargs)
val_batch_sampler = DynamicKBatchSampler(val_time_dataset, batch_size=batch_size, drop_last=True, k_update_callback=update_k_per_batch, shuffle=True)
val_time_loader = DataLoader(val_time_dataset, batch_sampler=val_batch_sampler)

# Way to load a dataset with a specific lead time
lead_time = max_lead_time
train_dataset = QGDataset(lead_time=lead_time,dataset_mode='train', **kwargs)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# Way to load a dataset with a specific lead time
val_dataset = QGDataset(lead_time=lead_time, dataset_mode='val', **kwargs)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------------------------------

def generate_ensemble_from_batch(model, n_ens=10, selected_loader = val_loader, sampler_fn=heun_sampler):
    # Need to choose batch_size such that batch_size*n_ens fits on GPU
    model.eval()

    previous, current, time_labels = next(iter(selected_loader))
    lead_time = time_labels[0].item()

    with torch.no_grad():
        previous = previous.to(device)
        current = current.to(device)
        previous_state = previous[:,:vars].repeat(n_ens, 1, 1, 1)

        class_labels = previous.repeat(n_ens, 1, 1, 1)
        time_labels = torch.ones(class_labels.shape[0], device=device, dtype=int) * lead_time / max_lead_time

        latents = torch.randn_like(previous_state, device=device)
        
        predicted_state = sampler_fn(model, latents, class_labels, time_labels, sigma_max=80, sigma_min=0.03, rho=7, num_steps=20, S_churn=2.5, S_min=0.75, S_max=80, S_noise=1.05)
        predicted_latent = predicted_state # previous_state +  * residual_scaling(torch.tensor(lead_time))
        predicted = predicted_latent
        
        predicted_unnormalized = renormalize(predicted)
        current_unnormalized = renormalize(current)

        predicted_unnormalized = predicted_unnormalized.view(n_ens, current.size(0), current.size(1), current.size(2), current.size(3))

    return predicted_unnormalized, current_unnormalized


def train():

    # -----------------------------------------------------

    #model_type = 'large' #'attention'
    input_times = (1 + len(initial_times))*vars + static_vars
    time_emb = 1

    if 'single' in model_choice:
        time_emb = 0
    
    if 'residual' in model_choice:
        model_type = 'large'

    
    model = EDMPrecond(filters=filters, img_channels=input_times, out_channels=vars, img_resolution = 64, time_emb=time_emb, 
                       model_type=model_type, sigma_data=1, sigma_min=0.02, sigma_max=88, label_dropout=label_dropout)
    loss_fn = WGCLoss(lat, lon, device, precomputed_std=precomputed_std)
    calculate_WRMSE = calculate_AreaWeightedRMSE(lat, lon, device).calculate
    calculate_WSkill = calculate_AreaWeightedRMSE(lat, lon, device).skill
    calculate_WSpread = calculate_AreaWeightedRMSE(lat, lon, device).spread

    print(name, flush=True)
    print(model_choice, flush=True)
    print("Num params: ", sum(p.numel() for p in model.parameters()), flush=True)
    model.to(device)

    print("Lead-times", lead_time_range, flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=1000)

    loss_values = []
    val_loss_values = []
    best_val_loss = float('inf')

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for previous, current, time_label in (val_time_loader):
            current = current.to(device)
            previous = previous.to(device)
            time_label = time_label.to(device)
            
            current_latent = (current)

            previous_state = previous[:,:vars]
            class_labels = previous
                                
            target_latent = (current_latent)# - previous_state) / residual_scaling(time_label[0])
            
            loss = loss_fn(model, target_latent, class_labels, time_label/max_lead_time, 1)
            
            total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_time_loader)

    print(f'Starting Validation Loss: {avg_val_loss:.4f}', flush=True)

    # Setup for logging
    log_file_path = result_path / f'training_log.csv'
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Average Training Loss', 'Validation Loss'])
    
    # Sampling
    val_dataset.set_lead_time(24*3)
    forecast, truth = generate_ensemble_from_batch(model, n_ens=10, selected_loader = val_loader, sampler_fn=heun_sampler)
    wrmse_1 = calculate_WRMSE(forecast, truth)
    skill_1 = calculate_WSkill(forecast, truth)
    spread_1 = calculate_WSpread(forecast, truth)

    val_dataset.set_lead_time(24*5)
    forecast, truth = generate_ensemble_from_batch(model, n_ens=10, selected_loader = val_loader, sampler_fn=heun_sampler)
    wrmse_2 = calculate_WRMSE(forecast, truth)
    skill_2 = calculate_WSkill(forecast, truth)
    spread_2 = calculate_WSpread(forecast, truth)

    for channel in range(wrmse_1.shape[-1]):  # Loop over the last dimension (channels)
        print(f'Channel {channel + 1}:')
        print(f'  3d - WRMSE: {np.mean(wrmse_1[..., channel].flatten()):.2f}, '
            f'Skill: {np.mean(skill_1[..., channel].flatten()):.2f}, '
            f'Spread: {np.mean(spread_1[..., channel].flatten()):.2f}')
        print(f'  5d - WRMSE: {np.mean(wrmse_2[..., channel].flatten()):.2f}, '
            f'Skill: {np.mean(skill_2[..., channel].flatten()):.2f}, '
            f'Spread: {np.mean(spread_2[..., channel].flatten()):.2f}', flush=True)
        best_val_sample = np.mean(wrmse_1 + wrmse_2)/2

    """
    Training starts here
    """

    """def calc_lead_time_range(u, lead_time_range):
        dt_min = 6
        dt_max = 12

        if u <= 0.1:
            dt_min = dt
            dt_max = dt

        elif u <= 0.2:
            dt_min = 2*dt
            dt_max = max_lead_time /5

        return new_lead_time_range"""
    

    def time_scaling(epoch, num_epochs):
        u = epoch / (num_epochs-1)
        return int(max(min(dt * (1 + (max_lead_time * np.sqrt(u)) // dt), max_lead_time), dt))

    dt_min = dt
    dt_max = dt

    for epoch in range(num_epochs):
        dt_min = dt_max
        dt_max = time_scaling(epoch, num_epochs)
        new_lead_time_range = [dt_min, dt_max, dt]
        
        # lead_time_range = [dt, int(dt * (1 + max_lead_time*epoch/num_epochs//dt)), dt]
        train_time_loader.dataset.set_lead_time_range(new_lead_time_range)
        print("Lead-time", new_lead_time_range, flush=True)

        model.train()  # Set model to training mode
        total_train_loss = 0

        for previous, current, time_label in (train_time_loader):
            current = current.to(device)
            previous = previous.to(device)
            time_label = time_label.to(device)
            
            optimizer.zero_grad()
            
            with torch.no_grad():
                current_latent = (current)
                previous_state = previous[:,:vars]
                class_labels = previous

                target_latent = (current_latent)# - previous_state) / residual_scaling(time_label[0])
                
            loss = loss_fn(model, target_latent, class_labels, time_label/max_lead_time, epoch/num_epochs)

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
                
                current_latent = (current)

                previous_state = previous[:,:vars]
                class_labels = previous
                                   
                target_latent = (current_latent)# - previous_state) / residual_scaling(time_label[0])
                
                loss = loss_fn(model, target_latent, class_labels, time_label/max_lead_time, epoch/num_epochs)
                
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

        ## Val
        val_dataset.set_lead_time(24*3)
        forecast, truth = generate_ensemble_from_batch(model, n_ens=10, selected_loader = val_loader, sampler_fn=heun_sampler)
        wrmse_1 = calculate_WRMSE(forecast, truth)
        skill_1 = calculate_WSkill(forecast, truth)
        spread_1 = calculate_WSpread(forecast, truth)

        val_dataset.set_lead_time(24*5)
        forecast, truth = generate_ensemble_from_batch(model, n_ens=10, selected_loader = val_loader, sampler_fn=heun_sampler)
        wrmse_2 = calculate_WRMSE(forecast, truth)
        skill_2 = calculate_WSkill(forecast, truth)
        spread_2 = calculate_WSpread(forecast, truth)

        for channel in range(wrmse_1.shape[-1]):  # Loop over the last dimension (channels)
            print(f'Channel {channel + 1}:')
            print(f'  3d - WRMSE: {np.mean(wrmse_1[..., channel].flatten()):.2f}, '
                f'Skill: {np.mean(skill_1[..., channel].flatten()):.2f}, '
                f'Spread: {np.mean(spread_1[..., channel].flatten()):.2f}')
            print(f'  5d - WRMSE: {np.mean(wrmse_2[..., channel].flatten()):.2f}, '
                f'Skill: {np.mean(skill_2[..., channel].flatten()):.2f}, '
                f'Spread: {np.mean(spread_2[..., channel].flatten()):.2f}', flush=True)
        
        best_val_sample = np.mean(wrmse_1 + wrmse_2)/2
        val_sample = np.mean(wrmse_1 + wrmse_2)/2

        # Checkpointing
        if val_sample < best_val_sample:
            best_val_sample = val_sample
            torch.save(model.state_dict(), result_path/f'best_sample_model.pth')

train()