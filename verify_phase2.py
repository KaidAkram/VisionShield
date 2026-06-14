import torch
import numpy as np
from PIL import Image

from model_loader import load_target_model, get_imagenet_labels
from utils.image_utils import get_preprocess_pipeline
from attacks.fgsm import fgsm_attack, pgd_attack

def main():
    print("--- VisionShield Phase 2 Verification ---")
    
    # 1. Setup Model and Data
    print("\n[1] Initializing model and generating synthetic input...")
    model = load_target_model()
    labels = get_imagenet_labels()
    
    # Create a random RGB image
    np.random.seed(42) # Deterministic for consistent report
    random_array = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    synthetic_image = Image.fromarray(random_array, 'RGB')
    
    preprocess = get_preprocess_pipeline()
    # Cache the original clean tensor to verify immutability later
    clean_tensor_ref = preprocess(synthetic_image).unsqueeze(0)
    tensor_input = clean_tensor_ref.clone().detach()
    
    # 2. Baseline Forward Pass
    print("\n[2] Performing Baseline Forward Pass...")
    with torch.no_grad():
        logits = model(tensor_input)
        baseline_probs = torch.nn.functional.softmax(logits, dim=1)
        
    baseline_prob, baseline_catid = torch.topk(baseline_probs, 1)
    baseline_class_idx = baseline_catid.item()
    baseline_conf = baseline_prob.item()
    
    print(f"    Baseline Prediction:")
    print(f"    - Class Index: {baseline_class_idx} ({labels[baseline_class_idx]})")
    print(f"    - Confidence: {baseline_conf * 100:.2f}%")
    
    # Target label is the predicted class (we want to perform an untargeted attack to make it misclassify)
    target_label = torch.tensor([baseline_class_idx])
    
    # Attack budgets
    epsilon = 0.05
    alpha = 0.01
    num_iter = 10
    
    # 3. FGSM Attack
    print(f"\n[3] Executing FGSM Attack (epsilon={epsilon})...")
    fgsm_adv_tensor = fgsm_attack(model, tensor_input, epsilon, target_label)
    
    with torch.no_grad():
        fgsm_logits = model(fgsm_adv_tensor)
        fgsm_probs = torch.nn.functional.softmax(fgsm_logits, dim=1)
        
    fgsm_prob, fgsm_catid = torch.topk(fgsm_probs, 1)
    fgsm_class_idx = fgsm_catid.item()
    fgsm_conf = fgsm_prob.item()
    
    print(f"    FGSM Prediction:")
    print(f"    - Class Index: {fgsm_class_idx} ({labels[fgsm_class_idx]})")
    print(f"    - Confidence: {fgsm_conf * 100:.2f}%")
    
    # 4. PGD Attack
    print(f"\n[4] Executing PGD Attack (epsilon={epsilon}, alpha={alpha}, num_iter={num_iter})...")
    pgd_adv_tensor = pgd_attack(model, tensor_input, epsilon, target_label, alpha=alpha, num_iter=num_iter)
    
    with torch.no_grad():
        pgd_logits = model(pgd_adv_tensor)
        pgd_probs = torch.nn.functional.softmax(pgd_logits, dim=1)
        
    pgd_prob, pgd_catid = torch.topk(pgd_probs, 1)
    pgd_class_idx = pgd_catid.item()
    pgd_conf = pgd_prob.item()
    
    print(f"    PGD Prediction:")
    print(f"    - Class Index: {pgd_class_idx} ({labels[pgd_class_idx]})")
    print(f"    - Confidence: {pgd_conf * 100:.2f}%")
    
    # 5. Verification Checks
    print("\n[5] Running Integrity Verification Checks...")
    
    fgsm_shape_match = list(fgsm_adv_tensor.shape) == [1, 3, 224, 224]
    pgd_shape_match = list(pgd_adv_tensor.shape) == [1, 3, 224, 224]
    print(f"    FGSM adversarial output dimensions match [1, 3, 224, 224]: {fgsm_shape_match}")
    print(f"    PGD adversarial output dimensions match [1, 3, 224, 224]: {pgd_shape_match}")
    assert fgsm_shape_match and pgd_shape_match, "Tensor shape mismatch after attacks!"
    
    # Verify original tensor was not mutated
    is_mutated = not torch.equal(tensor_input, clean_tensor_ref)
    print(f"    Original tensor mutated in place: {is_mutated}")
    assert not is_mutated, "Original tensor was mutated in place! Cloning/Detaching failed."
    
    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    main()
