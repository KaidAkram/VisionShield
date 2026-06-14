import torch
import torchvision.transforms.functional as TF

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def denormalize(tensor):
    """Reverses ImageNet normalization."""
    device = tensor.device
    return tensor * _STD.to(device) + _MEAN.to(device)

def normalize(tensor):
    """Applies ImageNet normalization."""
    device = tensor.device
    return (tensor - _MEAN.to(device)) / _STD.to(device)

def gaussian_smoothing_defense(tensor, kernel_size=3, sigma=1.0):
    """
    Applies Gaussian blur to a tensor to reduce high-frequency adversarial noise.
    Returns a cloned, defended tensor without altering the original graph.
    """
    # Clone and detach to ensure we don't alter computational graph or original tensor
    t = tensor.clone().detach()
    
    # Kernel size must be a list/tuple of two ints
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
        
    smoothed = TF.gaussian_blur(t, kernel_size, [sigma, sigma])
    return smoothed

def bit_depth_reduction_depth(tensor, bits=4):
    """
    Quantizes pixel data into 2^bits uniform color levels to step over 
    micro-perturbations. Operates on raw pixel space and renormalizes.
    """
    t = tensor.clone().detach()
    
    # Denormalize to get values in standard [0, 1] range
    t_raw = denormalize(t)
    t_raw = torch.clamp(t_raw, 0.0, 1.0)
    
    # Quantize to target bits
    num_bins = (2 ** bits) - 1
    t_quantized = torch.round(t_raw * num_bins) / num_bins
    
    # Renormalize back to ImageNet space
    t_renormalized = normalize(t_quantized)
    
    return t_renormalized
