import torch
import torch.nn as nn
import math
from diffusion_networks import *

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "GenCast: Diffusion-based
# ensemble forecasting for medium-range weather".
class GCLoss:
    def __init__(self, sigma_min=0.02, sigma_max=88, rho=7, sigma_data=1, time_noise=0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.time_noise = time_noise
    
    def __call__(self, net, images, class_labels=None, time_labels=None):
        # Time Augmentation
        if self.time_noise > 0:
            time_labels = time_labels + torch.randn_like(time_labels, device=images.device, dtype=torch.float32) * self.time_noise
        
        # Sample from F inverse
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        rho_inv = 1 / self.rho
        sigma_max_rho = self.sigma_max ** rho_inv
        sigma_min_rho = self.sigma_min ** rho_inv
        
        sigma = (sigma_max_rho + rnd_uniform * (sigma_min_rho - sigma_max_rho)) ** self.rho
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = images
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, class_labels, time_labels)
        loss = weight * ((D_yn - y) ** 2)
        loss = loss.sum().mul(1/images.shape[0])
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "GenCast: Diffusion-based
# ensemble forecasting for medium-range weather".
class WGCLoss:
    def __init__(self, lat, lon, device, sigma_min=0.02, sigma_max=88, rho=7, sigma_data=1, time_noise=0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.time_noise = time_noise
        self.area_weights = torch.tensor(comp_area_weights_simple(lat, lon), device=device, dtype=torch.float32)
    
    def __call__(self, net, images, class_labels=None, time_labels=None):
        # Time Augmentation
        if self.time_noise > 0:
            time_labels = time_labels + torch.randn_like(time_labels, device=images.device, dtype=torch.float32) * self.time_noise
        
        # Sample from F inverse
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        rho_inv = 1 / self.rho
        sigma_max_rho = self.sigma_max ** rho_inv
        sigma_min_rho = self.sigma_min ** rho_inv
        
        sigma = (sigma_max_rho + rnd_uniform * (sigma_min_rho - sigma_max_rho)) ** self.rho
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = images
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, class_labels, time_labels)
        loss = self.area_weights * weight * ((D_yn - y) ** 2)
        loss = loss.sum().mul(1/images.shape[0])
        return loss

#----------------------------------------------------------------------------
# Area weighted loss function from the codebase 
# diffusion-models-for-weather-prediction

class calculate_WeightedRMSE:
    def __init__(self, weights, device):
        self.weights = torch.tensor(weights, device=device, dtype=torch.float32)   

    def diff(self, input: torch.tensor, target: torch.tensor):
        return (self.weights * (input - target) ** 2)
    
    def loss_fn(self, input: torch.tensor, target: torch.tensor):
        return self.diff(input, target).mean().sqrt()
    
    def calculate(self, input: torch.tensor, target: torch.tensor):
        dims_to_include = list(range(2, input.dim()))
        return self.diff(input, target).mean(dim=dims_to_include).sqrt().cpu().detach().numpy()
    
    def spread(self, input: torch.tensor, target: torch.tensor):
        ens_mean = input.mean(dim=0)
        N = input.size(0)
        dims_to_include = list(range(1, target.dim()))
        spread = ((self.weights*(ens_mean - input)**2).sum(dim=0)/(N - 1)).mean(dim=dims_to_include).sqrt().cpu().detach().numpy()
        return spread



class calculate_AreaWeightedRMSE(calculate_WeightedRMSE):
    def __init__(self, lat, lon, device):
        super().__init__(weights=comp_area_weights_simple(lat, lon), device=device)
    

# ----

def comp_area_weights_simple(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Calculate the normalized area weights.

    Args:
        lat (np.ndarray): Array of latitudes of grid center points
        lon (np.ndarray): Array of longitudes of grid center points

    Returns:
        np.ndarray: 2D array of relative area sizes.
    """
    area_weights = np.cos(lat * np.pi / 180).reshape(-1, 1)
    area_weights = np.repeat(area_weights, lon.size, axis=1)
    area_weights *= (lat.size * lon.size / np.sum(area_weights))
    return area_weights

class WeightedLoss(torch.nn.Module):
    def __init__(self, weights, device):
        super().__init__()
        self.weights = torch.tensor(weights, device=device, dtype=torch.float32)

    def forward(self, input, target):
        raise NotImplementedError("Must be implemented by subclasses")

class WeightedMAELoss(WeightedLoss):
    def forward(self, input, target):
        return torch.mean(self.weights * torch.abs(input - target))

class WeightedMSELoss(WeightedLoss):
    def forward(self, input, target):
        return torch.mean(self.weights * (input - target) ** 2)

class AreaWeightedLoss(torch.nn.Module):
    def __init__(self, lat, lon, device, loss_cls):
        super().__init__()
        weights = comp_area_weights_simple(lat, lon)
        self.loss_instance = loss_cls(weights, device)

    def forward(self, input, target):
        return self.loss_instance(input, target)

class AreaWeightedMAELoss(AreaWeightedLoss):
    def __init__(self, lat, lon, device):
        super().__init__(lat, lon, device, WeightedMAELoss)

class AreaWeightedMSELoss(AreaWeightedLoss):
    def __init__(self, lat, lon, device):
        super().__init__(lat, lon, device, WeightedMSELoss)