import torch
import math

def mu_sigma(t, option):
    r""" Mean and Variance schedules
    Input: Time is fractional time
    Option: 0: Cos, 1: Linear
    """
    if option == 0:   
        # Using cos scheduling
        omega = math.acos(math.sqrt(0.001)) 
        # Get mean (1->0)
        mu = torch.cos(omega*t)**2
    elif option == 1:
        # Use linear scheduling
        # Get mean (1->0)
        mu = 1-t
    # Set mach_eps to prevent blow up of variance at t=0
    mach_eps = 1e-2
    # Get variance (0 -> 1)
    sigma = torch.sqrt(1-mu**2+mach_eps**2)
    return mu.float(), sigma.float()

def forward_sample(x,t, option):
    r""" 
    Forward sampler for VP-SDE
    Arguments:
        x: clean image (Size: B X W X H)
        t: sampling diffusion time
        Option: 0: Cos, 1: Linear
    """
    # Sample noise 
    eps = torch.randn_like(x)
    # Get mean, variance schedules
    mu, sigma = mu_sigma(t, option)
    
    mu   = mu.view(-1, 1, 1).cuda()
    sigma = sigma.view(-1, 1, 1).cuda()
    
    # Get sample of noisy image at current diffusion time
    x_out = mu*x + sigma*eps
    return x_out, eps


def get_score(pred,x_noisy,t, option):
    r""" 
    Gets score for VP-SDE based on NN that predicts noise
    Arguments:
        pred: predicted noise from NN
        x_noisy: noisy image (Size: B X W X H)
        t: diffusion time
        Option: 0: Cos, 1: Linear
    """   
    # Get mean, variance schedules
    _, sigma = mu_sigma(t, option)
    score =  - pred/(sigma)
    return score