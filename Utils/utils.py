import importlib
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import torch
import pickle
import shutil


def print_config(obj, indent=0):
    """Recursively prints all attributes of a class or object."""
    if not hasattr(obj, "__dict__") and not isinstance(obj, type):  # If it's a simple value, print it
        print(" " * indent + str(obj))
        return

    for attr_name in dir(obj):
        if attr_name.startswith("__"):  # Skip special attributes
            continue

        attr_value = getattr(obj, attr_name)

        if isinstance(attr_value, type):  # If it's a class, recurse into it
            print(" " * indent + f"{attr_name}:")
            print_config(attr_value, indent + 4)
        elif not callable(attr_value):  # Print regular attributes
            print(" " * indent + f"{attr_name} = {attr_value}")

def save_file(model, scheduler, optimizer, train_result, run_number, save_dir):
    ### Move all code files
    # Move config file to results folder
    source_config_file = f"/gdata/projects/ml_scope/Turbulence/Diffusion_V0001/Src/Config/Run{run_number:05d}.py"
    destination_directory = f"/gdata/projects/ml_scope/Turbulence/Diffusion_V0001/Results/Run{run_number:05d}/Code"
    os.makedirs(destination_directory, exist_ok=True)

    if os.path.exists(source_config_file):
        destination_config_file = os.path.join(destination_directory, f"Run{run_number:05d}.py")
        shutil.copy(source_config_file, destination_config_file)
    
    # Move remaining code
    source_directory = f"/gdata/projects/ml_scope/Turbulence/Diffusion_V0001/Src"

    # List of folders to ignore
    folders_to_ignore = [
        f"/gdata/projects/ml_scope/Turbulence/Diffusion_V0001/Src/Config"]
    # Move the configuration file to the destination directory
    for root, dirs, files in os.walk(source_directory):
        # Determine the relative path and the destination path
        relative_path = os.path.relpath(root, source_directory)
        destination_path = os.path.join(destination_directory, relative_path)
        # Check if the current directory should be ignored
        if any(ignored_dir in root for ignored_dir in folders_to_ignore):
            continue
        # Create the directory structure in the destination
        os.makedirs(destination_path, exist_ok=True)
        # Copy the files
        for file in files:
            source_file = os.path.join(root, file)
            destination_file = os.path.join(destination_path, file)
            shutil.copy(source_file, destination_file)
            
     ### Save ML model, scheduler and results
    # Save the model's state dictionary
    model_state_dict_path = os.path.join(save_dir, 'model_state_dict.pth')
    torch.save(model.state_dict(), model_state_dict_path)

    # Save the scheduler's state dictionary
    scheduler_state_dict_path = os.path.join(save_dir, 'scheduler_state_dict.pth')
    torch.save(scheduler.state_dict(), scheduler_state_dict_path)
    
    # Save the optimizer's state dictionary
    optimizer_state_dict_path = os.path.join(save_dir, 'optimizer_state_dict.pth')
    torch.save(optimizer.state_dict(), optimizer_state_dict_path)

    # Save the training result with train_losses and val_accs
    train_result_path = os.path.join(save_dir, 'train_result.pkl')
    with open(train_result_path, 'wb') as f:
        pickle.dump(train_result, f)       
    