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
import time

from utils import *
from autoencoder import Autoencoder
from diffusion_networks import *

# Defining the constants and configurations used throughout the notebook.
iterations = 2101000 # 101000
on_remote = True
data_path = Path(f'/nobackup/smhid20/users/sm_maran/dpr_data/simulations/QG_samples_SUBS_{iterations}.npy') if on_remote else Path(f'C:/Users/svart/Desktop/MEX/data/QG_samples_SUBS_{iterations}.npy')
k = 50
spinup = 1001
spacing = 1
p_train = 0.8
mean_data = 0.003394413273781538
std_data = 9.174626350402832

date = '2024-02-21'
autoencoder_model = 'ae-2ds-32f-1l-150e-L1-0wd-0.00001l1' #'ae-3ds-16f-2l-150e-L1-0wd-0.00001l1'# 

autoencoder_path = Path(f'/nobackup/smhid20/users/sm_maran/results/{date}/{autoencoder_model}/') if on_remote else Path(f'C:/Users/svart/Desktop/MEX/results/{date}/{autoencoder_model}/')

#ae-2ds-32f-1l-150e-L1-0wd-0.00001l1
mean_data_latent = -0.661827802658081
std_data_latent = 5.319980144500732
std_residual_latent = 5.724194526672363
std_residual = 1.0840884447097778

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = QGSamplesDataset(data_path, 'train', p_train, k, spinup, spacing, iterations, mean_data, std_data, device)
val_dataset = QGSamplesDataset(data_path, 'val', p_train, k, spinup, spacing, iterations, mean_data, std_data, device)
test_dataset = QGSamplesDataset(data_path, 'test', p_train, k, spinup, spacing, iterations, mean_data, std_data, device)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

import json
# FILEPATH: /c:/Users/svart/Desktop/MEX/QG-Forecasting/Diffusion.ipynb
# Load the saved model
saved_model = torch.load(autoencoder_path / 'best_model.pth')

# Read parameters from JSON file
with open(autoencoder_path / 'config.json', 'r') as json_file:
    parameters = json.load(json_file)

# Extract the desired parameters
filters = parameters['filters']
latent_dim = parameters['latent_dim']
no_downsamples = parameters['no_downsamples']

# Create an instance of the ConvolutionalAutoencoder class
autoencoder = Autoencoder(filters= filters, no_latent_channels=latent_dim, no_downsamples=no_downsamples)

# Load the state_dict of the saved model into the conv_autoencoder
autoencoder.load_state_dict(saved_model)

autoencoder.to(device)
autoencoder.eval()

print("Autoencoder loaded successfully!")

forecast = False
islatent = False

def train():
    optimizer = optim.Adam(model.parameters())
    loss_fn = GCLoss()

    loss_values = []
    val_loss_values = []
    best_val_loss = float('inf')

    num_epochs = 100

    # Setup for logging
    log_file_path = 'training_log.csv'
    with open(log_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Epoch', 'Average Training Loss', 'Validation Loss'])
    
    if islatent:
        # Assume first is latent
        if not forecast:

            # Define the model
            model = GCPrecond(filters=32, no_downsamples=2, img_channels=1, img_resolution = 16)
            model.to(device)

            # Training loop

            for epoch in range(num_epochs):
                start_time = time.time()

                model.train()  # Set model to training mode
                total_train_loss = 0

                for data, _ in train_loader:
                    img = data
                    optimizer.zero_grad()

                    with torch.no_grad():
                        latent = autoencoder.encoder(img)
                        latent = (latent - mean_data_latent) / std_data_latent
                                    
                    loss = loss_fn(model, latent)

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
                    for data,_ in val_loader:
                        img = data

                        latent = autoencoder.encoder(img)
                        latent = (latent - mean_data_latent) / std_data_latent
                        
                        loss = loss_fn(model, latent)
                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_loader)

                # Checkpointing
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), 'best_model.pth')

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

        else:
            model = GCPrecond(sigma_data=1, filters=32, no_downsamples=2, img_channels=2, img_resolution = 16)
            model.to(device)

            # Forecast residual


            for epoch in range(num_epochs):
                start_time = time.time()

                model.train()  # Set model to training mode
                total_train_loss = 0

                for previous, current in train_loader:        
                    optimizer.zero_grad()

                    with torch.no_grad():
                        current_latent = autoencoder.encoder(current)
                        previous_latent = autoencoder.encoder(previous)
                        
                        residual_latent = (current_latent - previous_latent) / std_residual_latent
                        
                    loss = loss_fn(model, residual_latent, previous_latent)

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
                        previous_latent = autoencoder.encoder(previous)
                        residual_latent = (current_latent - previous_latent) / std_residual_latent
                                        
                        loss = loss_fn(model, residual_latent, previous_latent)

                        total_val_loss += loss.item()
                        
                avg_val_loss = total_val_loss / len(val_loader)

                # Checkpointing
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), 'best_model.pth')

                val_time = time.time() - start_time
                start_time = time.time()
                # scheduler.step()
                    
                # Sample and plot image
                #(model
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

    else:
    # Assume is not latent
        if not forecast:

            # Define the model
            model = GCPrecond(filters=32, no_downsamples=2, img_channels=1, img_resolution = 65, isLatent=False)
            model.to(device)

            # Training loop

            for epoch in range(num_epochs):
                start_time = time.time()

                model.train()  # Set model to training mode
                total_train_loss = 0

                for data, _ in train_loader:
                    img = data
                    optimizer.zero_grad()

                    loss = loss_fn(model, img)

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
                    for data,_ in val_loader:
                        img = data

                        loss = loss_fn(model, img)
                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_loader)

                # Checkpointing
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), 'best_model.pth')

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

        else:
            model = GCPrecond(sigma_data=1, filters=32, no_downsamples=2, img_channels=2, img_resolution = 65, isLatent=False)
            model.to(device)

            # Forecast residual


            for epoch in range(num_epochs):
                start_time = time.time()

                model.train()  # Set model to training mode
                total_train_loss = 0

                for previous, current in train_loader:        
                    optimizer.zero_grad()

                    residual = (current - previous) / std_residual
                        
                    loss = loss_fn(model, residual, previous)

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
                        
                        residual = (current - previous) / std_residual
                     
                        loss = loss_fn(model, residual, previous)
                    total_val_loss += loss.item()
                        
                avg_val_loss = total_val_loss / len(val_loader)

                # Checkpointing
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), 'best_model.pth')

                val_time = time.time() - start_time
                start_time = time.time()
                # scheduler.step()
                    
                # Sample and plot image
                #(model
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

