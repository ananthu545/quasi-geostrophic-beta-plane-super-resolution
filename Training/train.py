import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append('/gdata/projects/ml_scope/Turbulence/Diffusion_V0001/Src/Utils/')
from Utils.dataclass import TrainResult
import sys
sys.path.append('/gdata/projects/ml_scope/Turbulence/Diffusion_V0001/Src/Models/')
from Models.sde import mu_sigma, forward_sample
import tqdm
import torch.optim as optim

## Base versions
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, lr, option, save_dir, condition):
    train_result = TrainResult(num_epochs, lr, train_losses = [], val_accs=[])

    for epoch in range(num_epochs):  
        total_train = 0
        model.train()

        for batch in train_loader:
            if condition:
                clean_image, diff_time, cond_image = batch
                cond_image = cond_image.float().cuda()
            else:
                clean_image, diff_time = batch
                cond_image=None            
            
            clean_image = clean_image.float().cuda()

            optimizer.zero_grad()

            mu, sigma = mu_sigma(diff_time, option)

            sigma = sigma.view(-1, 1, 1).cuda()
            mu = mu.view(-1, 1, 1).cuda()

            # Get noisy image from forward process
            noisy_image, eps = forward_sample(clean_image,diff_time, option)

            #Predicted noise
            output_pred = model(noisy_image,diff_time,cond_image)
            #pred_clean=(1/mu)*(noisy_image - sigma*output_pred)

            loss = criterion(output_pred, eps,diff_time)

            train_result.train_losses.append(loss.item())
            total_train += loss.item()

            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()

        total_train = total_train/len(train_loader)
        print(f"Train Epoch: {epoch}, Loss: {total_train}")

        scheduler.step()
        if epoch % 5 == 0:
            save_path = os.path.join(save_dir, f"model_{epoch:05d}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint at {save_path}")       

    return train_result