import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Tuple, Union, Optional, List
import sys
sys.path.append('/gdata/projects/ml_scope/Turbulence/Diffusion_V0001/Src/Models/')
from Models.sde import mu_sigma, forward_sample, get_score
#sys.path.append('/gdata/projects/ml_scope/Turbulence/QG_V0001/Src/Operators/')
#from Operators.spectral_conversion import to_physical, to_spectral


## Sampler assume VP-SDE, and contains flags to implement SDEdit, DPS, conditional, and CFG
def get_samples_corrected(x_init, option, x_cond, seed, num_samples, diffusion_time_steps, num_corrections, model_uncond, 
                          model_cond, corr_scale, sdedit, DPS, grid,spectral_operator, cond_flag, CFG):   
    r""" Sampling schemes
    x_init: Initial condition [W,H]
    option: Option for noise scheduling; 0: Cosine, 1: Linear
    x_cond: conditioning image (Observation) [W,H]
    seed: Seed for random generation
    num_samples: number of samples generated (Ensemble size)
    diffusion_time_steps: Number of discretized diffusion pseudo-time-steps
    num_correction: Number of Langevin Monte Carlo corrections
    model_uncond: Unconditional model (NN)
    model_cond: Conditional model (NN)
    corr_scale: Langevin correction scale (used to calculate delta)
    sdedit.flag: Uses SDEdit (changes IC & time to t_i)
    sdedit.time: t_i to start SDEdit
    DPS.flag: uses DPS (need to provide measurement operator)
    DPS.scale: Low-res coarsening scale
    grid: Low-res grid operator
    spectral_operator: Low-res spectral operators
    DPS.C: Guidance strength for DPS
    DPS.sigma_measure: Noise in measurement
    DPS.gappy: Uses sparse and gappy observations
    cond_flag: Uses conditional input
    CFG.flag: Uses classifier-free guidance  
    CFG.weight: Guidance strength for CFG
    Output: [num_samples,W,H,diffusion_time_steps+1]
    """  
    ## Set seed
    torch.manual_seed(seed)
    
    ## Initialization
    if sdedit.flag:
        # Use SDEdit initialization
        x=x_cond.repeat([num_samples,1,1]) #[num_samples,W,H]
        t_i = (sdedit.time/diffusion_time_steps)*torch.ones(1).cuda()
        x,_=forward_sample(x,t_i,option)
        loop_init=diffusion_time_steps-sdedit.time+1
    else:
        # Use regular initialization
        x=torch.randn_like(x_init.repeat([num_samples,1,1])) #[num_samples,W,H]
        t_i = torch.ones(1).cuda()
        loop_init=1
    
    ## Initialize save trajectory
    N, W, H = x.shape
    T = diffusion_time_steps
    traj = torch.zeros(N, W, H, T+1, device=x.device)
    traj[..., 0] = x
    
    x_cond = x_cond.repeat([num_samples,1,1])

    # Get noise schedule
    mu_i, sigma_i = mu_sigma(t_i, option)
    
    # Start reverse process
    with torch.no_grad():
    
        for step_idx, i in enumerate(range(loop_init, T+1), start=1): 
            
            t_i_dash = torch.tensor(1-i/diffusion_time_steps).cuda()

            # Get mean (1->0)
            mu_i_dash, sigma_i_dash = mu_sigma(t_i_dash,option)

            r =  mu_i_dash/mu_i

            if not cond_flag:
                ## Use unconditional model
                pred = model_uncond(x,t_i)
            elif CFG.flag:
                ## USE CFG weighting
                pred= model_uncond(x,t_i) + CFG.weight*(model_cond(x,t_i,x_cond)-model_uncond(x,t_i))
            else:
                ## Use conditional model
                pred=model_cond(x,t_i,x_cond)


            if DPS.flag:
                ## Use DPS score
                if i != diffusion_time_steps:
                    x_i_1 = r*x + (r - sigma_i_dash/sigma_i)*(sigma_i**2)*(get_score(pred,x,t_i,option)+
                                                                           DPS_score(x,x_cond,t_i, model_uncond,option,
                                                                                     DPS,grid,spectral_operator))
                else:
                    x_i_1 = x + (sigma_i**2)*(get_score(pred,x,t_i,option)+DPS_score(x,x_cond,t_i, model_uncond,option,
                                                                                     DPS,grid,spectral_operator))
            else:
                ## Use regular score
                if i != diffusion_time_steps:
                    x_i_1 = r*x + (r - sigma_i_dash/sigma_i)*(sigma_i**2)*get_score(pred,x,t_i,option)
                else:
                    x_i_1 = x + (sigma_i**2)*(get_score(pred,x,t_i,option))

            ## Langevin Monte Carlo Corrections
            for _ in range(num_corrections):
                z = torch.randn_like(x)
                if not cond_flag:
                    ## Use unconditional model
                    pred = model_uncond(x_i_1,t_i_dash)
                elif CFG.flag:
                    ## USE CFG weighting
                    pred= model_uncond(x_i_1,t_i_dash) + CFG.weight*(model_cond(x_i_1,t_i_dash,x_cond)-model_uncond(x_i_1,t_i_dash))
                else:
                    ## Use unconditional model
                    pred=model_cond(x_i_1,t_i_dash,x_cond)
                    
                if DPS.flag:
                    eps = get_score(pred,x_i_1,t_i_dash,option)+DPS_score(x_i_1,x_cond,t_i_dash,model_uncond,
                                                                          option,DPS,grid,spectral_operator)
                else:
                    eps = get_score(pred,x_i_1,t_i_dash,option)
                        
                #delta = corr_scale / F.mse_loss(eps,0*eps)
                
                denom = max(F.mse_loss(eps, torch.zeros_like(eps)), 1e-8)
                delta = corr_scale / denom
                x_i_1 = x_i_1 +   delta * eps + torch.sqrt(2*delta)*z


            x = x_i_1
            mu_i = mu_i_dash
            sigma_i =  sigma_i_dash
            t_i = t_i_dash 
            traj[..., step_idx] = x
                      
    return traj

######## Observation operators

def DPS_score(x_noisy,x_measured, diff_time, model_uncond, option, DPS,grid,spectral_operator):
    with torch.enable_grad():
        x_in = x_noisy.clone().detach().requires_grad_(True)
        
        mu, sigma = mu_sigma(diff_time, option)
        
        eps_pred = model_uncond(x_in,diff_time)
        
        x_cleaned = (1/mu)*(x_in - sigma*eps_pred)
                
        noisy_measured, sigma_scaled =  A_les_interleaved(x_cleaned,DPS,diff_time, grid,
                                                          spectral_operator, option)      
        measure_error =  F.mse_loss(noisy_measured, x_measured,reduction='none') # Dims [B X H X W]
        
        measure_error = torch.mean(measure_error, dim=[1,2]) # Dims [B]
                
        temp_score = torch.autograd.grad(outputs=measure_error,inputs=x_in,grad_outputs=torch.ones_like(measure_error))[0] # Dims [B x H x W]
        # Final measurement matching score term for DPS        
        dps_score= -(0.5/sigma_scaled)* temp_score 
                
    return dps_score.clone().detach()

def A_les_interleaved(x_in,DPS,diff_time, grid, spectral_operator, option): 
    r""" 
    x_in: Noisy image (IC) [B,W,H]
    DPS.scale: Low-res coarsening scale
    diff_time: Diffusion timestep
    grid: Low-res grid operator
    spectral_operator: Low-res spectral operators
    diffusion_time_steps: Number of discretized time steps
    DPS.C: Guidance strength for DPS
    DPS.sigma_measure: Noise in measurement
    DPS.gappy: Use gappy observations
    option: option for noise scheduling
    Output: data_les: coarsened field, sigma_scaled: DPS score weighting factor
    """ 
    
    qh=torch.fft.rfftn(x_in.squeeze(), dim=(-2,-1), norm="forward")
    #qh=to_spectral(x_in.squeeze()) ##Incorrect usage, takes FFT over batch too
    
    G = torch.exp(-4*spectral_operator.krsq*((DPS.scale*grid.dx)**2)/24)

    qh_dr = G*qh
    G = spectral_operator.ky<(torch.pi/(DPS.scale*grid.dx))
    qh_dr = G*qh_dr
    G =  spectral_operator.kr<(torch.pi/(DPS.scale*grid.dx))
    q_dr = torch.fft.irfftn(G*qh_dr, dim=(-2,-1), norm="forward")
    #q_dr = to_physical(G*qh_dr)  ##Incorrect usage, takes FFT over batch too  
    
    if q_dr.dim() == 2:
        # single image [H, W]
        x = q_dr.unsqueeze(0).unsqueeze(0)          # → [1, 1, H, W]
        pooled = F.avg_pool2d(x, kernel_size=DPS.scale, stride=DPS.scale)
        pooled = pooled.squeeze(0).squeeze(0)      # → [H/scale, W/scale]
        data_les = pooled.repeat_interleave(DPS.scale, dim=0) \
                         .repeat_interleave(DPS.scale, dim=1)  # → [H, W]

    elif q_dr.dim() == 3:
        # batch of images [B, H, W]
        x = q_dr.unsqueeze(1)                      # → [B, 1, H, W]
        pooled = F.avg_pool2d(x, kernel_size=DPS.scale, stride=DPS.scale)
        pooled = pooled.squeeze(1)                 # → [B, H/scale, W/scale]
        data_les = pooled.repeat_interleave(DPS.scale, dim=1) \
                         .repeat_interleave(DPS.scale, dim=2)  # → [B, H, W]
        
    else:
        raise ValueError(f"Unsupported tensor shape {q_dr.shape!r}; expected 2D or 3D.")
        
    ## Sparse and gappy observations (two swaths with a gap)    
    if DPS.gappy:
        frac   = 0.2
        H, W   = data_les.shape[-2], data_les.shape[-1]
        mask_w = int(W * frac)
        mid_start, mid_end = (W - mask_w)//2, (W - mask_w)//2 + mask_w
        left_start, left_end = 0, mask_w
        right_start = W - mask_w

        data_les[..., left_start:left_end]  = 0
        data_les[..., mid_start:mid_end]    = 0
        data_les[..., right_start:W]        = 0
        

    mu, sigma = mu_sigma(diff_time, option) 
                
    sigma_scaled =  DPS.sigma_measure**2 + DPS.C*(sigma/mu)**2    
    
    return data_les, sigma_scaled