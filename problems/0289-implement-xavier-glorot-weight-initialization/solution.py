import torch
import torch.nn as nn

def xavier_initialization(shape: tuple, mode: str = 'uniform', seed: int = None) -> torch.Tensor:
    """
    Implement Xavier/Glorot weight initialization using PyTorch.
    
    Args:
        shape: Tuple of (fan_in, fan_out) representing weight matrix dimensions
        mode: 'uniform' or 'normal' initialization
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Initialized weight matrix as torch.Tensor
    """
    fan_in, fan_out = shape
    weight = torch.empty(shape)
    
    if mode == 'uniform':
        limit = (6.0 / (fan_in + fan_out)) ** 0.5
        weight.uniform_(-limit, limit)
        
    elif mode == 'normal':
        sigma = (2.0 / (fan_in + fan_out)) ** 0.5
        weight.normal_(0, sigma)
        
    else:
        raise ValueError(f"Unsupported mode: {mode}. Expected 'uniform' or 'normal'.")
        
    return weight