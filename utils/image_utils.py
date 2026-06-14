import torch
import torchvision.transforms as transforms
import numpy as np

def get_preprocess_pipeline():
    """
    Returns a torchvision.transforms.Compose pipeline that resizes images to 256,
    center-crops to 224, converts to tensor, and normalizes using ImageNet statistics.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def denormalize_tensor(tensor):
    """
    Takes a normalized tensor, reverses the ImageNet scale factors,
    switches dimensions to Channels-Last (HWC), clamps values strictly between 0.0 and 1.0,
    and returns a NumPy array suitable for visual rendering.
    """
    # Clone tensor to avoid modifying original
    t = tensor.clone().detach().cpu()
    
    # Remove batch dimension if present
    if t.dim() == 4:
        t = t.squeeze(0)
        
    # Reverse normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    t = t * std + mean
    
    # Clamp strictly between 0.0 and 1.0
    t = torch.clamp(t, 0.0, 1.0)
    
    # Switch to Channels-Last (HWC)
    t = t.permute(1, 2, 0)
    
    return t.numpy()
