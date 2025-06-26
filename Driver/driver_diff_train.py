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

# Parse the command-line arguments
args = parser.parse_args()
run_number = args.run_num

## Load and import the config file
config_module = f'Config.Run{run_number:05d}'

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

### Load data
from Dataloaders.dataloader import DiffusionDataset

# Set default values for some attributes
config.params.data.field_std = getattr(config.params.data, 'field_std', 1)
config.params.data.cond_data_file_prefix = getattr(config.params.data, 'cond_data_file_prefix', None)

# Creating dataset objects
train_dataset = DiffusionDataset(config.params.data.data_dir,config.params.data.file_prefix,config.params.data.run_start,
                                 config.params.data.run_end,config.params.data.min_t,
                                 config.params.sde.time_steps,config.params.data.field_std,config.params.data.cond_data_file_prefix)

# Creating dataloaders
train_loader = DataLoader(train_dataset, batch_size=config.params.data.batch_size, shuffle=True)
       
print(f"Dataloaders created successfully")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

# Load model, criterion and optimizer
from Models.networks import UNet_large

model = UNet_large(channels=config.params.network.channels,in_channels=config.params.network.in_channels,
                   out_channels=config.params.network.out_channels,dropout_rate=config.params.network.dropout_rate,
                  attention=config.params.network.attention,condition=config.params.network.condition).cuda()

if config.params.network.condition:
    print(f"Conditioning enabled")
else:
    print(f"Conditioning disabled")

if config.params.network.attention:
    print(f"Attention enabled")
else:
    print(f"Attention disabled")

from Loss_functions.loss import EDM_loss
criterion = EDM_loss(sigma_data=0.5)

optimizer = optim.AdamW(model.parameters(), lr=config.params.train.learning_rate)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.params.train.num_epochs)

print(f"Model, criterion and optimizer initialized successfully")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

### Begin training
from Training.train import train_model
torch.cuda.empty_cache()
save_dir = f'/gdata/projects/ml_scope/Turbulence/Diffusion_V0001/Results/Run{run_number:05d}/Checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
train_result = train_model(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                           num_epochs=config.params.train.num_epochs, lr=config.params.train.learning_rate, 
                           option=config.params.sde.option, save_dir = save_dir, condition = config.params.network.condition)
      
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))

## Save directory
save_dir = f'/gdata/projects/ml_scope/Turbulence/Diffusion_V0001/Results/Run{run_number:05d}/Checkpoints'
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

## Save and copy codes, weights, scheduler and optimizer dicts
from Utils.utils import save_file
save_file(model=model, scheduler=scheduler, optimizer=optimizer, train_result=train_result, run_number=run_number,save_dir=save_dir)
print(f"Files saved successfully")
now = datetime.datetime.now()
print(now.strftime("%Y-%m-%d %H:%M:%S"))