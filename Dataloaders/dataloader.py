import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import itertools
import os

class DiffusionDataset(Dataset):
    """ Dataset and dataloader for diffusion models
    Output: Data_t: size B X W X H
    diffusion_time/self.diffusion_time_steps: diffusion pseudo-time b/w 0 and 1  
    """
    def __init__(self, data_dir,file_prefix,run_start,run_end,min_t,diffusion_time_steps, field_std = 1, cond_file_prefix=None):
        # data_dir:    File path for ensemble data
        # file_prefix: File name for ensemble data
        # run_start:  Start of run numbers
        # run_end :   End of run numbers
        # min_t:     Timestep to start training after spin up (after self-similiarity is reached)
        # diffusion_time_steps: Number of diffusion pseudo-time-steps
        # field_std: Standard deviation of the DNS fields for normalization
        # cond_file_prefix: File name for conditional field
        
        
        self.data_dir = data_dir
        self.file_prefix=file_prefix
        self.run_start = run_start
        self.run_end = run_end
        self.min_t = min_t
        self.diffusion_time_steps = diffusion_time_steps
        self.field_std = field_std
        self.cond_file_prefix = cond_file_prefix
        
        # List to accumulate data from each run
        all_runs_data = []
        # If conditional training
        if self.cond_file_prefix is not None:
            all_runs_data_cond=[]
        
        for run_id in range(run_start,run_end+1):       
            file_path = os.path.join(self.data_dir,f"Run{run_id:05d}",f"{self.file_prefix}{run_id:05d}.npy")
            data = np.load(file_path, mmap_mode='r') # Size [H,W,T,C]
            # Discard time steps prior to min_t and choose only vorticity (0th index)
            data = data[:,:,min_t:,0].squeeze()
            all_runs_data.append(data)
            # If conditional training
            if self.cond_file_prefix is not None:
                file_path = os.path.join(self.data_dir,f"Run{run_id:05d}",f"{self.cond_file_prefix}{run_id:05d}.npy")
                data_cond = np.load(file_path, mmap_mode='r') # Size [H,W,T,C]
                # Discard time steps prior to min_t and choose only vorticity (0th index)
                data_cond = data_cond[:,:,min_t:,0].squeeze()
                all_runs_data_cond.append(data_cond)
            
        self.ensemble_data = np.stack(all_runs_data, axis=-1) # Size [H,W,T,run_num]
        # If conditional training
        if self.cond_file_prefix is not None:
            self.ensemble_data_cond = np.stack(all_runs_data_cond, axis=-1) # Size [H,W,T,run_num]
        # Creat indices for random sampling in the dataloader
        n_runs = self.ensemble_data.shape[3]
        n_times = self.ensemble_data.shape[2]
        self.ensemble_indices = list(itertools.product(range(n_runs), range(n_times),range(self.diffusion_time_steps+1)))

    def __len__(self):
        return len(self.ensemble_indices)

    def __getitem__(self, idx):
        run_index, time, diffusion_time = self.ensemble_indices[idx]

        # Fetching data
        data_t = (self.ensemble_data[:, :, time,run_index]/self.field_std).copy() # Size [H,W,T,run_num]
        
        # If conditional training
        if self.cond_file_prefix is not None:
            data_t_cond = (self.ensemble_data_cond[:, :, time,run_index]/self.field_std).copy() # Size [H,W,T,run_num]
            return data_t, diffusion_time/self.diffusion_time_steps, data_t_cond

        return data_t, diffusion_time/self.diffusion_time_steps