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
