import torch
import numpy as np
from tqdm import tqdm


def calculate_RMSE(predicted, truth):
    dims_to_include = list(range(2, predicted.dim()))
    return torch.sqrt(torch.mean((predicted - truth) ** 2, dim=dims_to_include)).cpu().detach().numpy()

def calculate_climatology(selected_loader):
    mean = 0
    count = 0
    with torch.no_grad():
        for _, current in tqdm(selected_loader):
            mean += torch.sum(current, dim=0)
            count += current.size(0)
    mean = mean / count
    return mean

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

    a = (forecast - truth).norm(dim=0, p=1)
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
