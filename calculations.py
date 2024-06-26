import torch
import numpy as np
from tqdm import tqdm


def calculate_RMSE(predicted, truth):
    dims_to_include = list(range(2, predicted.dim()))
    return torch.sqrt(torch.mean((predicted - truth) ** 2, dim=dims_to_include)).cpu().detach().numpy()

def calculate_MAE(predicted, truth):
    dims_to_include = list(range(2, predicted.dim()))
    return torch.mean(torch.abs(predicted - truth), dim=dims_to_include).cpu().detach().numpy()

def calculate_skill_and_spread_score(forecast, truth):
    ens_mean = forecast.mean(dim=0)
    dims_to_include = list(range(1, truth.dim()))

    skill = ((ens_mean - truth)**2).mean(dim=dims_to_include).sqrt()
    N = forecast.size(0)

    spread = (((ens_mean - forecast)**2).sum(dim=0)/(N - 1)).mean(dim=dims_to_include).sqrt()

    ratio = np.sqrt((N+1)/N) * spread / skill
    return skill.cpu().detach().numpy(), spread.cpu().detach().numpy(), ratio.cpu().detach().numpy()

def calculate_CRPS(forecast, truth):
    dims_to_include = list(range(1, truth.dim()))
    
    a = (forecast - truth).abs().mean(dim=0)
    b = (forecast.unsqueeze(1) - forecast.unsqueeze(0)).abs().mean(dim=(0,1)) * 0.5
    return (a - b).mean(dim=dims_to_include).cpu().detach().numpy()

def calculate_brier_score(forecast, truth, threshold=10):
    dims_to_include = list(range(1, truth.dim()))

    px = (forecast > threshold).to(torch.float32).mean(dim=0)
    py = (truth > threshold).to(torch.float32)
    
    brier = ((px - py)**2).mean(dim=dims_to_include)
    return brier.cpu().detach().numpy()

def calculate_covtrace(forecast):
    covtrace = np.zeros(forecast.size(1))
    for i in range(forecast.size(1)):
        forecast_i = forecast[:,i]
        covtrace[i] = torch.trace(torch.cov(forecast_i.view(forecast_i.size(0), -1)))

    return covtrace

def calculate_psnr(forecast, truth):
    mse = torch.mean((truth - forecast)**2, axis = (2,3,4))
    max_truth, _ = (truth**2).view(truth.shape[0],-1).max(dim=1)
    psnr = torch.mean(10 * torch.log10(max_truth/mse), axis=0)
    return psnr.cpu().detach().numpy()