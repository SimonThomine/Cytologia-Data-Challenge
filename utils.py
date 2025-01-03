import random
import numpy as np
import torch


def set_seed(seed):
    """Function to set the seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 


def get_device():
    if torch.cuda.is_available():  # Nvidia GPU
        device = "cuda"
    elif torch.backends.mps.is_available():  # Apple silicon
        device = "mps"
    else:
        device = "cpu"
    return device
