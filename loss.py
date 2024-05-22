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
# Area weighted loss function from the codebase 
# diffusion-models-for-weather-prediction
class WeightedMSELoss:
    def __init__(self, weights):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.weights = torch.tensor(weights, device=device)   

    def loss_fn(self, input: torch.tensor, target: torch.tensor):
        return (self.weights * (input - target) ** 2).mean()

class AreaWeightedMSELoss(WeightedMSELoss):
    def __init__(self, lat, lon):
        super().__init__(weights=comp_area_weights_simple(lat, lon))
    


def comp_area_weights_simple(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """An easier way to calculate the (already normalized) area weights.

    Args:
        lat (np.ndarray): Array of latitudes of grid center points
        lon (np.ndarray): Array of lontigutes of grid center points

    Returns:
        np.ndarray: 2d array of relative area sizes.
    """
    area_weights = np.cos(lat * (2 * np.pi) / 360)
    area_weights = area_weights.reshape(-1, 1).repeat(lon.shape[0],axis=-1)
    area_weights = (lat.shape[0]*lon.shape[0] / np.sum(area_weights)) * area_weights
    return area_weights
