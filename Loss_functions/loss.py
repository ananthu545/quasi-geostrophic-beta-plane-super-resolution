import torch
import torch.nn.functional as F
import torch.nn as nn

class EDM_loss(torch.nn.Module):
    def __init__(self, sigma_data=0.5):
        super(EDM_loss, self).__init__()
        self.sigma_data = sigma_data

    def forward(self, pred_state, target_state,t):
        r"""
        Arguments:
            pred_state: predicted clean image (Size: B X W X H)
            target_state: clean image
            t: diffusion pseudo-timestep
        """
        ## Uncomment if using EDM loss from Karras 2022.
        #_, sigma = mu_sigma(t)
        #sigma = sigma.view(-1, 1, 1).cuda()
        #scale = (sigma**2+sigma_data**2)/((sigma*sigma_data)**2)
        
        loss = F.mse_loss(pred_state, target_state)
            
        return  loss