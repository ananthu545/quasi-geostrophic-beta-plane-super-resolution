import dataclasses
import torch.nn as nn 
from typing import Tuple, Union, Optional, List

@dataclasses.dataclass
class TrainResult:
    r"""
    A collection containing everything we need to know about the training results
    """
    num_epochs: int
        
    lr: float

    # Training loss (saved at each iteration in `train_epoch`)
    train_losses: List[float]

    # Validation accuracies, before training and after each epoch
    val_accs: List[float]
