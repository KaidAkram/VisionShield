import torch
import numpy as np
from PIL import Image
from model_loader import load_target_model, get_imagenet_labels
from utils.image_utils import get_preprocess_pipeline, denormalize_tensor

def main():
    print("--- VisionShield Phase 1 Verification ---")
    
    # 1. Generate synthetic 3-channel RGB image
    print("[1] Generating synthetic image...")
    # Create a random RGB image of size 300x300
    random_array = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    synthetic_image = Image.fromarray(random_array, 'RGB')
    
    # 2. Feed it through preprocessing pipeline
    print("[2] Running preprocessing pipeline...")
    preprocess = get_preprocess_pipeline()
    tensor_input = preprocess(synthetic_image)
    
    # Add batch dimension
    tensor_input = tensor_input.unsqueeze(0)
    print(f"    Verified Tensor Dimensions: {tensor_input.shape}")
    assert list(tensor_input.shape) == [1, 3, 224, 224], "Tensor shape format does not match [1, 3, 224, 224]!"
    
    # 3. Verify Parameter Freezing
    print("[3] Loading and verifying ResNet-18 model...")
    model = load_target_model()
    
    frozen = all(not p.requires_grad for p in model.parameters())
    print(f"    Parameters Frozen (requires_grad=False): {frozen}")
    assert frozen, "Not all model parameters are frozen!"
    
    # 4. Forward Pass
    print("[4] Executing forward pass...")
    labels = get_imagenet_labels()
    
    with torch.no_grad():
        logits = model(tensor_input)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
    top_prob, top_catid = torch.topk(probabilities, 1)
    
    top_prob_value = top_prob.item()
    top_catid_value = top_catid.item()
    predicted_label = labels[top_catid_value]
    
    print(f"    Sample Classification Output:")
    print(f"    - Predicted Class Index: {top_catid_value}")
    print(f"    - Predicted Label: {predicted_label}")
    print(f"    - Confidence Rating: {top_prob_value * 100:.2f}%")
    
    # 5. Verify Denormalization
    print("[5] Verifying denormalization function...")
    denorm_img = denormalize_tensor(tensor_input)
    print(f"    Denormalized Image Shape: {denorm_img.shape}")
    print(f"    Denormalized Value Range: [{denorm_img.min():.4f}, {denorm_img.max():.4f}]")
    
    print("--- Verification Complete ---")

if __name__ == "__main__":
    main()
