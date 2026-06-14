import torch
import numpy as np
from PIL import Image

from model_loader import load_target_model, get_imagenet_labels
from utils.image_utils import get_preprocess_pipeline
from attacks.fgsm import pgd_attack
from defenses.smoothing import gaussian_smoothing_defense, bit_depth_reduction_depth

def main():
    print("--- VisionShield Phase 3 Verification ---")
    
    # 1. Setup
    print("\n[1] Initializing model and generating synthetic input...")
    model = load_target_model()
    labels = get_imagenet_labels()
    
    np.random.seed(42)
    random_array = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    synthetic_image = Image.fromarray(random_array, 'RGB')
    
    preprocess = get_preprocess_pipeline()
    clean_tensor = preprocess(synthetic_image).unsqueeze(0)
    
    # Baseline
    with torch.no_grad():
        baseline_logits = model(clean_tensor)
        baseline_probs = torch.nn.functional.softmax(baseline_logits, dim=1)
        
    baseline_prob, baseline_catid = torch.topk(baseline_probs, 1)
    baseline_class_idx = baseline_catid.item()
    
    print(f"    Baseline Prediction: {baseline_class_idx} ({labels[baseline_class_idx]}) - {baseline_prob.item()*100:.2f}%")
    
    target_label = torch.tensor([baseline_class_idx])
    
    # 2. PGD Attack
    print("\n[2] Running PGD Attack (epsilon=0.05) to generate adversarial tensor...")
    adv_tensor = pgd_attack(model, clean_tensor, epsilon=0.05, target_label=target_label, alpha=0.01, num_iter=10)
    
    with torch.no_grad():
        adv_logits = model(adv_tensor)
        adv_probs = torch.nn.functional.softmax(adv_logits, dim=1)
        
    adv_prob, adv_catid = torch.topk(adv_probs, 1)
    adv_class_idx = adv_catid.item()
    
    print(f"    Adversarial Prediction: {adv_class_idx} ({labels[adv_class_idx]}) - {adv_prob.item()*100:.2f}%")
    
    # 3. Gaussian Smoothing Defense
    print("\n[3] Applying Gaussian Smoothing Defense (kernel_size=5, sigma=1.0)...")
    gaussian_defended = gaussian_smoothing_defense(adv_tensor, kernel_size=5, sigma=1.0)
    
    with torch.no_grad():
        gauss_logits = model(gaussian_defended)
        gauss_probs = torch.nn.functional.softmax(gauss_logits, dim=1)
        
    gauss_prob, gauss_catid = torch.topk(gauss_probs, 1)
    gauss_class_idx = gauss_catid.item()
    
    print(f"    Gaussian Defended Prediction: {gauss_class_idx} ({labels[gauss_class_idx]}) - {gauss_prob.item()*100:.2f}%")
    
    # 4. Bit-Depth Reduction Defense
    print("\n[4] Applying Bit-Depth Reduction Defense (bits=4)...")
    bit_defended = bit_depth_reduction_depth(adv_tensor, bits=4)
    
    with torch.no_grad():
        bit_logits = model(bit_defended)
        bit_probs = torch.nn.functional.softmax(bit_logits, dim=1)
        
    bit_prob, bit_catid = torch.topk(bit_probs, 1)
    bit_class_idx = bit_catid.item()
    
    print(f"    Bit-Depth Defended Prediction: {bit_class_idx} ({labels[bit_class_idx]}) - {bit_prob.item()*100:.2f}%")
    
    # 5. Dimension check
    print("\n[5] Running Integrity Verification Checks...")
    gauss_shape = list(gaussian_defended.shape)
    bit_shape = list(bit_defended.shape)
    
    print(f"    Gaussian tensor output dimensions match [1, 3, 224, 224]: {gauss_shape == [1, 3, 224, 224]}")
    print(f"    Bit-depth tensor output dimensions match [1, 3, 224, 224]: {bit_shape == [1, 3, 224, 224]}")
    assert gauss_shape == [1, 3, 224, 224]
    assert bit_shape == [1, 3, 224, 224]

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    main()
