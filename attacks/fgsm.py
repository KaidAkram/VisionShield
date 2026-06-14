import torch
import torch.nn as nn

# Define standard ImageNet bounds for clamping normalized tensors back to valid pixel ranges
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def get_normalized_bounds(device):
    """
    Computes the valid lower and upper bounds for a normalized tensor
    so that when denormalized, the values stay within [0.0, 1.0].
    """
    lower_bound = (0.0 - _MEAN.to(device)) / _STD.to(device)
    upper_bound = (1.0 - _MEAN.to(device)) / _STD.to(device)
    return lower_bound, upper_bound

def fgsm_attack(model, tensor, epsilon, target_label, criterion=nn.CrossEntropyLoss()):
    """
    Computes a single-step gradient sign update (Fast Gradient Sign Method).
    Returns the adversarial tensor.
    
    This is an untargeted attack. By adding the gradient of the loss w.r.t the true label,
    we maximize the loss, pushing the model to misclassify.
    """
    # Clone tensor to prevent in-place mutation and enable gradient tracking
    input_tensor = tensor.clone().detach().to(tensor.device)
    input_tensor.requires_grad = True

    # Forward pass
    outputs = model(input_tensor)
    
    # Calculate loss
    target_label = target_label.to(tensor.device)
    loss = criterion(outputs, target_label)
    
    # Zero gradients, compute backward pass to calculate gradients
    model.zero_grad()
    loss.backward()
    
    # Collect the sign of the data gradient
    data_grad = input_tensor.grad.data
    sign_data_grad = data_grad.sign()
    
    # Create the perturbed image (untargeted attack -> add gradient to maximize loss)
    perturbed_tensor = input_tensor + epsilon * sign_data_grad
    
    # Clamp to ensure validity in raw pixel space bounds
    lower_bound, upper_bound = get_normalized_bounds(tensor.device)
    perturbed_tensor = torch.max(torch.min(perturbed_tensor, upper_bound), lower_bound)
    
    return perturbed_tensor.detach()

def pgd_attack(model, tensor, epsilon, target_label, alpha=2/255, num_iter=10, criterion=nn.CrossEntropyLoss()):
    """
    Computes an iterative L_infinity bounded adversarial perturbation (Projected Gradient Descent).
    Every iteration clips the tensor back into the epsilon-neighborhood of the original clean tensor,
    and clamps values to standard bounds.
    """
    original_tensor = tensor.clone().detach().to(tensor.device)
    perturbed_tensor = original_tensor.clone().detach()
    lower_bound, upper_bound = get_normalized_bounds(tensor.device)
    target_label = target_label.to(tensor.device)

    for i in range(num_iter):
        perturbed_tensor.requires_grad = True
        
        # Forward pass
        outputs = model(perturbed_tensor)
        loss = criterion(outputs, target_label)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Collect gradient sign
        data_grad = perturbed_tensor.grad.data
        sign_data_grad = data_grad.sign()
        
        # Iterate: update the perturbed image
        adv_tensor = perturbed_tensor + alpha * sign_data_grad
        
        # Clip perturbation to epsilon neighborhood of original tensor (L_infinity projection)
        eta = torch.clamp(adv_tensor - original_tensor, min=-epsilon, max=epsilon)
        perturbed_tensor = original_tensor + eta
        
        # Clamp to normalized pixel bounds to ensure validity
        perturbed_tensor = torch.max(torch.min(perturbed_tensor, upper_bound), lower_bound).detach()
        
    return perturbed_tensor
