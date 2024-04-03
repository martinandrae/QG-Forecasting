import torch
import torch.nn as nn
import math
from edm_networks import *

"""Model architectures and preconditioning schemes used in the paper"""

#----------------------------------------------------------------------------
# Simple Unet
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1)
        #time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        freqs = torch.arange(half_dim, device=device)

        embeddings = math.log(10000) / (half_dim - 1)
        
        embeddings = torch.exp(freqs * - embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self, filters, no_downsamples, image_channels, time_emb_dim, isLatent=True):
        super().__init__()

        down_channels = [filters * 2**i for i in range(no_downsamples + 1)]
        up_channels = list(reversed(down_channels))
        out_channels = image_channels//2

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        kernel_size = 3 if isLatent else 4 # Change when changing from 65x65
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], kernel_size, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])
        
        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.ConvTranspose2d(up_channels[-1], out_channels, kernel_size, padding=1)

    def forward(self, x, timestep, class_labels=None):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        
        if class_labels is not None:
            x = torch.cat((x, class_labels), dim=1)
        
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)


# ------------------------------------------------------------------------------
# Model
    

# ------------------------------------------------------------------------------
# Preconditioner
class GCPrecond(torch.nn.Module):
    def __init__(self, sigma_data=1, sigma_min=0.02, sigma_max=88, filters=128, model = 'ncsnpp', img_channels=1, img_resolution = 65, time_emb_dim=128, isLatent=True, time_emb=0):
        super(GCPrecond, self).__init__()
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        if model == 'ddpmpp':
            self.model = SongUNet(img_resolution=img_resolution, in_channels=img_channels, out_channels=1, \
                                embedding_type='positional', encoder_type='standard', decoder_type='standard', \
                                channel_mult_noise=1, resample_filter=[1,1], model_channels=filters, channel_mult=[2,2,2],
                                num_blocks=4,  time_emb=time_emb)
        elif model == 'ncsnpp':
            self.model = TimeSongUNet(img_resolution=img_resolution, in_channels=img_channels, out_channels=1, \
                                embedding_type='fourier', encoder_type='residual', decoder_type='standard', \
                                channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=filters, channel_mult=[2,2,2], \
                                time_emb=time_emb)
        elif model == 'simple':
            self.model = SimpleUnet(filters=filters, no_downsamples=2, image_channels=img_channels, time_emb_dim=time_emb_dim, isLatent=isLatent)
        
        else:
            raise ValueError('Model not recognized')
    
        #self.model = SimpleUnet(filters=filters, no_downsamples=no_downsamples, image_channels=self.img_channels, time_emb_dim=time_emb_dim, isLatent=isLatent)
        #self.model = DhariwalUNet(img_resolution=img_resolution, in_channels=img_channels, out_channels=1, model_channels=192, channel_mult=[1,2,3,4])
        #self.model = CuboidTransformerUNet(input_shape=(img_channels, img_resolution, img_resolution, batch_size), target_shape=(1, img_resolution, img_resolution, batch_size))
        
    def forward(self, x, sigma, class_labels=None, time_labels=None):
        dtype = torch.float32
        x = x.to(dtype) # EMA does this
        sigma = sigma.to(dtype).reshape(-1, 1, 1, 1) # EMA does this

        # Change these if we want
        c_skip = self.sigma_data ** 2 / (sigma **2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2+ self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x).to(dtype), c_noise.flatten(), class_labels, time_labels)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(dtype)
        
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    

# ------------------------------------------------------------------------------
# Loss function
class GCLoss:
    def __init__(self, sigma_min=0.02, sigma_max=88, rho=7, sigma_data=1, time_noise=0.25):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.time_noise = time_noise
    
    def __call__(self, model, images, class_labels=None, time_labels=None):
        # Sample from F inverse
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        rho_inv = 1 / self.rho
        sigma_max_rho = self.sigma_max ** rho_inv
        sigma_min_rho = self.sigma_min ** rho_inv
        
        sigma = (sigma_max_rho + rnd_uniform * (sigma_min_rho - sigma_max_rho)) ** self.rho

        # Loss weight
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        # Generate noisy images
        noise = torch.randn_like(images)
        noisy_images = images + sigma * noise

        # Time
        time_labels = time_labels.to(images.device) + torch.randn_like(time_labels, device=images.device, dtype=torch.float32) * self.time_noise

        # Forward pass
        denoised_images = model(noisy_images, sigma, class_labels, time_labels)

        # Compute loss
        loss = weight * ((denoised_images - images) ** 2)

        return loss.sum().mul(1/images.shape[0])
    
    