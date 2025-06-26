import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Tuple, Union, Optional, List

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SelfAttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # B x (H*W) x C'
        proj_key   = self.key(x).view(B, -1, H * W)                     # B x C' x (H*W)
        energy     = torch.bmm(proj_query, proj_key)                    # B x (H*W) x (H*W)
        attention  = self.softmax(energy)                               # B x (H*W) x (H*W)

        proj_value = self.value(x).view(B, -1, H * W)                   # B x C x (H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))         # B x C x (H*W)
        out = out.view(B, C, H, W)                                      # B x C x H x W

        out = self.gamma * out + x
        return out

class ResBlock(nn.Module):
    # Remember all of these have circular padding for periodic BCs
    def __init__(self, in_channels, out_channels, dropout_rate=0):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular')
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular')
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = F.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        out += residual
        return F.relu(out)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, padding_mode='circular')
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(F.relu(self.conv(x)))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(F.relu(self.conv(x)))

class UNet_large(nn.Module):
    def __init__(self, channels=[32,64,128,256],in_channels=1, out_channels=1, dropout_rate=0, attention = False, condition = False):
        super(UNet_large, self).__init__()
        
        self.attention_enabled = attention
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.res_blocks = nn.ModuleList()
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Encoder: build downsampling layers
        prev_ch = in_channels
        for ch in channels:
            self.down_layers.append(DownsampleBlock(prev_ch, ch, dropout_rate))
            prev_ch = ch
            
        # Residual blocks at the bottom
        self.res_blocks.append(ResBlock(prev_ch, prev_ch, dropout_rate))
        self.res_blocks.append(ResBlock(prev_ch, prev_ch, dropout_rate))   
        
        # Optional self-attention at bottleneck
        if self.attention_enabled:
            self.self_attention = SelfAttentionBlock(prev_ch)
                   
        # Decoder: build upsampling layers (reverse order)
        for i in reversed(range(len(channels) - 1)):
            if i == len(channels) - 2:
                self.up_layers.append(UpsampleBlock(prev_ch, channels[i], dropout_rate))
            else:
                self.up_layers.append(UpsampleBlock(2*prev_ch, channels[i], dropout_rate))      
            prev_ch = channels[i]
            
        self.up_layers.append(UpsampleBlock(2*prev_ch,prev_ch, dropout_rate))
        
        self.final_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        

    def forward(self, x, t, x_cond = None):
        
        x_in = (x).float().unsqueeze(1) # Unsqueeze to match dims of CNN since we have no channels [B x 1 X W X H]
        
        if x_cond is not None:
            x_cond = x_cond.float().unsqueeze(1) #[B x 1 X W X H]
            x_in=torch.cat([x_in, x_cond], dim=1) #[B x 2 X W X H]
        
        # Encoder pass
        downs = []
        for down in self.down_layers:
            x_in = down(x_in)
            downs.append(x_in)

        # Residual blocks
        r = x_in
        for res in self.res_blocks:
            r = res(r)
        
        # Apply self-attention if enabled
        if self.attention_enabled:
            r = self.self_attention(r)
                   
        # Decoder pass with skip connections
        for i, up in enumerate(self.up_layers[:-1]):
            r = up(r)
            skip = downs[-(i + 2)]  # align skip connections
            r = torch.cat([r, skip], dim=1)
            
        # Final upsample (no skip)
        r = self.up_layers[-1](r)
        
        output = self.final_conv(r).squeeze()

        return output