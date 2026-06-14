import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights

def load_target_model():
    """
    Loads a pre-trained ResNet-18 model from torchvision.models using modern weights APIs,
    forces the model into evaluation mode, and explicitly sets requires_grad = False
    on all parameters to freeze them for downstream adversarial evaluation.
    """
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    
    model.eval()
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model

def get_imagenet_labels():
    """
    Fetches the 1,000 ImageNet text labels so model prediction indices 
    can be converted to human-readable strings.
    """
    weights = ResNet18_Weights.DEFAULT
    meta = weights.meta
    categories = meta["categories"]
    return categories
