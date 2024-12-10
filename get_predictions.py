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
import gc
import zarr

from utils import *
from loss import *
from sampler import *



# -----------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------

# Load configuration from a JSON file
def load_config(json_file):
    with open(json_file, 'r') as file:
        config = json.load(file)
    return config

# Setup argument parser to accept a JSON config file path
parser = argparse.ArgumentParser(description='Run model with configuration from JSON file.')
parser.add_argument('config_path', type=str, help='Path to JSON configuration file.')
args = parser.parse_args()

config_path = args.config_path
config = load_config(config_path)

# Constants and configurations loaded from JSON
name        = config['name']
model_path = config['model']
spacing     = config['spacing']
batch_size = config['batch_size']
t_max = config['t_max']
t_direct = config['t_direct']
t_min = t_direct
t_iter = config['t_iter']
n_ens = config['n_ens']
total_ens = config['total_ens']
job_index = config['job_index']
date = config['date']
#eps0 = config['eps']

total_ens = n_ens
date = '2024-09-29'

print(name, flush=True)
print("Job index", job_index)
print("[t_direct, t_iter, t_max]", [t_direct, t_iter, t_max],  flush=True)
print("n_ens:", n_ens,  flush=True)

# -----------------------------------------------------

# Path to the dataset, changes based on the execution environment

result_path = Path(f'/proj/berzelius-2022-164/users/sm_maran/results/predictions/{date}/{name}')
subd = '/proj/berzelius-2022-164/users/sm_maran/data/wb'


# Check if the directory exists, and create it if it doesn't
if not result_path.exists():
    result_path.mkdir(parents=True, exist_ok=True)

# Copy the JSON configuration file to the results directory
config_file_name = Path(config_path).name
shutil.copy(config_path, result_path / "config.json")

# -----------------------------------------------------

model_result_path = f'/proj/berzelius-2022-164/users/sm_maran/results/'


config_path = f'{model_result_path}/{model_path}/config.json'
config = load_config(config_path)

# Constants and configurations loaded from JSON
filters     = config['filters']
max_lead_time = config['max_lead_time']
label_dropout = config['label_dropout']
initial_times = config['initial_times']
dt = config['dt']
model_choice = config['model']

if t_iter > max_lead_time:
    print(f"The iterative lead time {t_iter} is larger than the max_lead_time {max_lead_time}")
if t_direct < dt:
    print(f"The direct lead time {t_direct} is smaller than dt {dt}")


# -----------------------------------------------------
var_names = ['z500', 't850', 't2m', 'u10', 'v10']
print(var_names,  flush=True)

#fname= 'z500_t850_1979-2018_5.625deg.npy'
fname= 'z500_t850_t2m_u10_v10_1979-2018_5.625deg.npy'

static_data_path = f'{subd}/orog_lsm_1979-2018_5.625deg.npy'
static_vars = 2

# Load the normalization factors from the JSON file
json_file = f'{subd}/norm_factors.json'
with open(json_file, 'r') as f:
    statistics = json.load(f)

mean_data = torch.tensor([stats["mean"] for (key, stats) in statistics.items() if key in var_names])
std_data = torch.tensor([stats["std"] for (key, stats) in statistics.items() if key in var_names])

norm_factors = np.stack([mean_data, std_data], axis=0)

vars = len(mean_data)

# -----------------------------------------------------

lead_time_range = [t_min, t_max, t_direct]

random_lead_time = 0 # Yes for random lead time
lead_time_max = 240

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
            'max_lead_time':    lead_time_max,
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

def interpolate_latents(alpha, lat1, lat2):
    return (1 - alpha**2).sqrt() * lat1 + alpha * lat2

def save_ensembles(model, n_ens, selected_loader, sampler_fn=heun_sampler):
    model.eval()

    # Initialize the dimensions based on the first batch
    previous, current, time_labels = next(iter(selected_loader))
    n_samples = current.shape[0]
    n_times = time_labels.shape[1]
    dx = current.shape[2]
    dy = current.shape[3]    
    # Total number of samples in the loader (you may need to update this to the correct size)
    total_samples = len(selected_loader.dataset)

    #start_ens = job_index * n_ens
    #end_ens = start_ens + n_ens

    # Create memory-mapped file for predictions
    #predictions_mmap = np.memmap(f'{result_path}/{name}.npy', dtype='float32', mode='w+', 
    #                             shape=(total_samples, n_ens, n_times, vars, dx, dy))
    
    predictions_mmap = zarr.open(f'{result_path}/{name}.zarr', mode='w', shape=(total_samples, total_ens, n_times, vars, dx, dy), 
                                 chunks = (1, total_ens, n_times, vars, dx, dy),
                                 dtype='float32', overwrite=True)


    iterative_steps = t_iter

    start_idx = 0  # Track index for where to write in the file

    for previous, current, time_labels in tqdm(selected_loader):        

        n_samples = current.shape[0]
        n_times = time_labels.shape[1]
        dx = current.shape[2]
        dy = current.shape[3]

        n_conditions = previous.shape[1]

        with torch.no_grad():
            previous = previous.to(device)
            current = current.view(-1, vars, dx, dy).to(device)
            
            direct_time_labels = torch.tensor(np.array([x for x in time_labels[0] if x <= iterative_steps]), device=device)
            n_iter = time_labels.shape[1] // direct_time_labels.shape[0]
            
            n_direct = direct_time_labels.shape[0]

            #previous_state = previous[:,:vars]
            class_labels = previous.repeat_interleave(n_direct * n_ens, dim=0) # Can not be changed if batchsz > 1

            static_fields = class_labels[:, -static_vars:]

            latent_shape = (n_samples * n_ens, vars, dx, dy)
            latents = torch.randn(latent_shape, device=device) # NEW

            direct_time_labels = direct_time_labels.repeat(n_ens * n_samples) # Can not be changed if n_direct > 1

            # Test
            predicted_combined = torch.zeros((n_samples, n_ens, n_times, vars, dx, dy), device=device)
            #eps = torch.tensor(eps0, device=device)

            for i in range(n_iter):
                latents = torch.randn(latent_shape, device=device)
                latents = latents.repeat_interleave(n_direct, dim=0) # Can not be changed if batchsz > 1 or n_ens >1
                """ # NEW
                latents_zeros = torch.zeros((n_direct, n_samples*n_ens, vars, dx, dy), device=device) # new
                latents_zeros[0] = latents
                
                for j in range(1, n_direct):
                    latents_eps = torch.randn_like(latents, device=device)
                    latents = interpolate_latents(eps, latents, latents_eps)
                    latents_zeros[j] = latents.clone()
                
                latents = torch.tensor(latents_zeros.cpu().numpy().reshape((n_direct*n_samples* n_ens, vars, dx, dy), order='F'), device=device)
                #"""

                predicted = sampler_fn(model, latents, class_labels, direct_time_labels / lead_time_max, 
                                        sigma_max=80, sigma_min=0.03, rho=7, num_steps=20, S_churn=2.5, S_min=0.75, S_max=80, S_noise=1.05)

                predicted_combined[:, :, i*n_direct:(i+1)*n_direct] = predicted.view(n_samples, n_ens, n_direct, vars, dx, dy)

                predicted = predicted.view(n_samples*n_ens, n_direct, vars, dx, dy)
                class_labels = class_labels.view(n_samples*n_ens, n_direct, n_conditions, dx, dy)[:, 0]

                if n_direct == 1:
                    class_labels = torch.cat((predicted[:,-1], class_labels[:,:vars]), dim=1)#.repeat_interleave(n_direct, dim=0)
                else:
                    class_labels = torch.cat((predicted[:,-1], predicted[:,-2]), dim=1).repeat_interleave(n_direct, dim=0) # Can not be changed if batchsz > 1
                
                if static_vars != 0:
                    class_labels = torch.cat((class_labels, static_fields), dim=1)

                #latents_eps = torch.randn(latent_shape, device=device)              # NEW
                #latents = interpolate_latents(eps, latents_zeros[-1], latents_eps)  # NEW

            # Save predictions incrementally to memory-mapped file
            predictions_mmap[start_idx:start_idx + n_samples, :, :, :, :, :] = renormalize(predicted_combined).view(n_samples, n_ens, n_times, vars, dx, dy).cpu().numpy()

            start_idx += n_samples

        gc.collect()
        torch.cuda.empty_cache()
        
    # Flush the memory-mapped file to ensure data is written
    #predictions_mmap.flush()


def evaluate():
    input_times = (1 + len(initial_times))*vars + static_vars
    time_emb = 1

    if 'single' in model_choice:
        time_emb = 0

    if 'residual' in model_choice:
        model_type = 'large'


    model = EDMPrecond(filters=filters, img_channels=input_times, out_channels=vars, img_resolution = 64, time_emb=time_emb, 
                        model_type=model_type, sigma_data=1, sigma_min=0.02, sigma_max=88, label_dropout=label_dropout)
    
    model.load_state_dict(torch.load(f'{model_result_path}/{model_path}/best_model.pth'))
    model.to(device)
    print("Loaded model", model_path,  flush=True)


    k_series = t_min + t_direct * np.arange(0, 1 + (t_max-t_min)//t_direct)
    val_time_series_dataset = QGDataset(lead_time=k_series, dataset_mode='test', **kwargs)
    val_time_series_loader = DataLoader(val_time_series_dataset, batch_size=batch_size, shuffle=False)

    sampler_fn = heun_sampler

    print(f"Datset contains {len(val_time_series_dataset)} samples",  flush=True)
    print(f"We do {len(val_time_series_loader)} batches",  flush=True)

    save_ensembles(model, n_ens=n_ens, selected_loader = val_time_series_loader, sampler_fn=sampler_fn)
    print("Done", flush=True)

evaluate()