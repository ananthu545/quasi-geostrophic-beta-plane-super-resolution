import json
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import datetime
import numpy as np
import IPython.display as display
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional, List
from tqdm.notebook import tqdm
import torch.optim as optim
import dataclasses
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import os
import sys
import warnings
import importlib

warnings.filterwarnings("ignore", category=UserWarning)
    
sys.path.append('/gdata/projects/ml_scope/Turbulence/Diffusion_V0001/Src')

# Create an argument parser
parser = argparse.ArgumentParser(description='Parse args')

# Add the run_num argument to specify the run number
parser.add_argument('--run_num', type=int, help='Run number for configuration file', required=True)
parser.add_argument('--gen_num', type=int, help='Forecast number for configuration file', required=True)

# Parse the command-line arguments
args = parser.parse_args()
run_number = args.run_num
gen_number=args.gen_num

## Load config file
config_module = f'Config.Run{run_number:05d}_Gen{gen_number:05d}'

from Utils.utils import print_config

try:
    config = importlib.import_module(config_module)
    print(f"Successfully loaded configuration from {config_module}")
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    print_config(config.params) 
except ModuleNotFoundError:
    print(f"Configuration file for run {run_num} not found.")
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    raise

# Set default values for some attributes
config.params.data.field_std = getattr(config.params.data, 'field_std', 1)
config.params.data.cond_data_file_prefix = getattr(config.params.data, 'cond_data_file_prefix', None)    
    
# Load model, criterion and optimizer
from Models.networks import UNet_large

model_cond = UNet_large(channels=config.params.network.channels,in_channels=2,
                   out_channels=config.params.network.out_channels,dropout_rate=config.params.network.dropout_rate,
                  attention=config.params.network.attention,condition=True).cuda()

model_uncond = UNet_large(channels=config.params.network.channels,in_channels=1,
                   out_channels=config.params.network.out_channels,dropout_rate=config.params.network.dropout_rate,
                  attention=config.params.network.attention,condition=False).cuda()

cond_model_load_directory = f"/gdata/projects/ml_scope/Turbulence/Diffusion_V0001/Results/Run{config.params.gen.model_cond:05d}/Checkpoints/model_{config.params.gen.checkpoint:05d}.pt"

uncond_model_load_directory = f"/gdata/projects/ml_scope/Turbulence/Diffusion_V0001/Results/Run{config.params.gen.model_uncond:05d}/Checkpoints/model_{config.params.gen.checkpoint:05d}.pt"

cond_model_load_dir = os.path.abspath(cond_model_load_directory)
uncond_model_load_dir = os.path.abspath(uncond_model_load_directory)

cond_model_state_dict = torch.load(cond_model_load_dir, map_location=torch.device('cpu'))
uncond_model_state_dict = torch.load(uncond_model_load_dir, map_location=torch.device('cpu'))

# Load the model weights
model_cond.load_state_dict(cond_model_state_dict)
model_uncond.load_state_dict(uncond_model_state_dict)

torch.cuda.empty_cache()
import gc
gc.collect()

print(f"Model loaded successfully")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# Store pseudo-spectral grids and spectral derivative operators
sys.path.append('/gdata/projects/ml_scope/Turbulence/QG_V0001/Src/')
from Grid.grid import Grid
grid_fDNS=Grid(Lx=2*math.pi,Ly=2*math.pi,Nx=64,Ny=64, device='cuda')

from Operators.operators import SpectralDerivatives
spec_der_fDNS = SpectralDerivatives(grid_fDNS)

### Load data
data_dir="/gdata/projects/ml_scope/Turbulence/QG_V0001/Results/"
run_id=config.params.gen.run_number
file_path = os.path.join(data_dir,f"Run{run_id:05d}", f"{config.params.data.file_prefix}{run_id:05d}.npy")
data = np.load(file_path, mmap_mode='r') # Size [H,W,T,C]
cond_file_path = os.path.join(data_dir,f"Run{run_id:05d}", f"{config.params.data.cond_data_file_prefix}{run_id:05d}.npy")
cond_data = np.load(cond_file_path, mmap_mode='r') # Size [H,W,T,C]
print(f"Data loaded successfully")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))
from Models.samplers import get_samples_corrected, DPS_score

# Initialize list to store output trajectory
all_trajs = [] 

if config.params.gen.snapshot is None:
    # Generate for all snapshots in the dataset
    t_start = config.params.data.min_t
    t_end = data.shape[2]
else:
    # Generate for only the specified snapshot
    t_start = config.params.data.min_t+config.params.gen.snapshot
    t_end = config.params.data.min_t+config.params.gen.snapshot+1

for time_index in range(t_start,t_end):
    data_t= (torch.from_numpy(data[:, :, time_index, 0].squeeze()).float().cuda()) / config.params.data.field_std
    cond_data_t = (torch.from_numpy(cond_data[:,:,time_index,0].squeeze()).float().cuda()) / config.params.data.field_std

    trajectory = get_samples_corrected(data_t,config.params.sde.option,cond_data_t,config.params.gen.seed,config.params.gen.num_samples,
                                       config.params.sde.time_steps,config.params.gen.num_corrections,model_uncond,model_cond,
                                       config.params.gen.corr_scale,config.params.gen.sdedit,config.params.gen.DPS,
                                       grid_fDNS,spec_der_fDNS,config.params.gen.cond_flag, config.params.gen.CFG)
    
    all_trajs.append(trajectory.unsqueeze(-1))

trajectory = torch.cat(all_trajs, dim=-1).squeeze()
    
## Save directory
save_dir = f'/gdata/projects/ml_scope/Turbulence/Diffusion_V0001/Results/Run{run_number:05d}/Gen{gen_number:05d}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

## Save as np file
trajectory_np = trajectory.detach().cpu().numpy()
np.save(os.path.join(save_dir, f'trajectory_Run{run_number:05d}_Gen{gen_number:05d}.npy'), trajectory_np)

print(f"Files saved successfully")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))