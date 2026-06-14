import streamlit as st
import torch
import numpy as np
from PIL import Image
import cv2

from model_loader import load_target_model, get_imagenet_labels
from utils.image_utils import get_preprocess_pipeline, denormalize_tensor
from attacks.fgsm import fgsm_attack, pgd_attack
from defenses.smoothing import gaussian_smoothing_defense, bit_depth_reduction_depth

# Set Streamlit page config
st.set_page_config(page_title="VisionShield AI", page_icon="🛡️", layout="wide")

@st.cache_resource
def get_model():
    return load_target_model()

@st.cache_data
def get_labels():
    return get_imagenet_labels()

def generate_synthetic_image():
    """Generates a reproducible synthetic noise sample when no image is provided."""
    np.random.seed(42)
    random_array = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    return Image.fromarray(random_array, 'RGB')

def predict(tensor, model):
    """Executes a forward pass and returns top prediction index and probability."""
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
    prob, catid = torch.topk(probs, 1)
    return catid.item(), prob.item()

def main():
    st.title("🛡️ VisionShield AI: Red Teaming & Defense")
    st.markdown("""
    **VisionShield AI** is an interactive framework for evaluating Convolutional Neural Networks (ResNet-18) against 
    white-box adversarial perturbations and demonstrating robust preprocessing mitigations.
    """)
    
    # Load Model and Utilities
    model = get_model()
    labels = get_labels()
    preprocess = get_preprocess_pipeline()
    
    # Sidebar: Target Configuration
    st.sidebar.header("🎯 Target Configuration")
    st.sidebar.text("Model: ResNet-18")
    is_frozen = all(not p.requires_grad for p in model.parameters())
    st.sidebar.success(f"Parameters Frozen: {is_frozen}")
    
    # Sidebar: Attack Configuration
    st.sidebar.header("⚔️ Attack Configuration")
    attack_type = st.sidebar.selectbox("Attack Type", ["FGSM", "PGD"])
    epsilon = st.sidebar.slider("Epsilon (Perturbation Budget)", min_value=0.0, max_value=0.1, value=0.03, step=0.005)
    
    pgd_iter = 10
    pgd_alpha = 0.01
    if attack_type == "PGD":
        pgd_iter = st.sidebar.slider("PGD Iterations", min_value=1, max_value=50, value=10)
        pgd_alpha = st.sidebar.slider("PGD Alpha", min_value=0.001, max_value=0.05, value=0.01, step=0.001)
        
    # Sidebar: Defense Configuration
    st.sidebar.header("🛡️ Defense Configuration")
    defense_type = st.sidebar.selectbox("Defense Mechanism", ["None", "Gaussian Smoothing", "Bit-Depth Reduction"])
    
    kernel_size = 3
    sigma = 1.0
    bits = 4
    if defense_type == "Gaussian Smoothing":
        kernel_size = st.sidebar.slider("Kernel Size (Must be odd)", min_value=3, max_value=11, value=5, step=2)
        sigma = st.sidebar.slider("Sigma", min_value=0.1, max_value=5.0, value=1.0)
    elif defense_type == "Bit-Depth Reduction":
        bits = st.sidebar.slider("Bits per Channel", min_value=1, max_value=8, value=4)
        
    # Main Panel: Data Input
    st.header("1. Input Data")
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        source_img = Image.open(uploaded_file).convert("RGB")
    else:
        st.info("No image uploaded. Using a synthetic noise sample.")
        source_img = generate_synthetic_image()
        
    # Preprocessing
    clean_tensor = preprocess(source_img).unsqueeze(0)
    
    # Baseline Prediction
    base_idx, base_prob = predict(clean_tensor, model)
    base_label = labels[base_idx]
    
    # Attack Execution
    target_label = torch.tensor([base_idx])
    
    if attack_type == "FGSM":
        adv_tensor = fgsm_attack(model, clean_tensor, epsilon, target_label)
    else:
        adv_tensor = pgd_attack(model, clean_tensor, epsilon, target_label, alpha=pgd_alpha, num_iter=pgd_iter)
        
    adv_idx, adv_prob = predict(adv_tensor, model)
    adv_label = labels[adv_idx]
    
    # Defense Execution
    if defense_type == "Gaussian Smoothing":
        defended_tensor = gaussian_smoothing_defense(adv_tensor, kernel_size=kernel_size, sigma=sigma)
    elif defense_type == "Bit-Depth Reduction":
        defended_tensor = bit_depth_reduction_depth(adv_tensor, bits=bits)
    else:
        defended_tensor = adv_tensor.clone().detach()
        
    def_idx, def_prob = predict(defended_tensor, model)
    def_label = labels[def_idx]
    
    # Visual Analysis
    st.header("2. Visual Analysis")
    col1, col2, col3 = st.columns(3)
    
    clean_vis = denormalize_tensor(clean_tensor)
    adv_vis = denormalize_tensor(adv_tensor)
    def_vis = denormalize_tensor(defended_tensor)
    
    # Compute absolute difference (amplified for visibility)
    diff = np.abs(adv_vis - clean_vis)
    diff_amplified = np.clip(diff * 10, 0, 1.0) 
    
    with col1:
        st.subheader("Clean Input")
        st.image(clean_vis, use_container_width=True)
        st.metric(label=f"Class: {base_label}", value=f"{base_prob*100:.2f}%")
        
    with col2:
        st.subheader("Perturbation (x10)")
        st.image(diff_amplified, use_container_width=True)
        delta_str = f"{(adv_prob - base_prob)*100:.2f}%" if adv_idx == base_idx else "Misclassified"
        st.metric(label=f"Class: {adv_label}", value=f"{adv_prob*100:.2f}%", delta=delta_str, delta_color="inverse")
        
    with col3:
        st.subheader(f"Defended ({defense_type})")
        st.image(def_vis, use_container_width=True)
        st.metric(label=f"Class: {def_label}", value=f"{def_prob*100:.2f}%")
        
    # Analytical Report
    st.header("3. Analytical Report")
    
    col_a, col_b, col_c = st.columns(3)
    col_a.info(f"**Baseline Prediction:**  \n{base_label} ({base_prob*100:.2f}%)")
    col_b.error(f"**Adversarial Prediction:**  \n{adv_label} ({adv_prob*100:.2f}%)")
    
    if defense_type != "None":
        if def_idx == base_idx:
            col_c.success(f"**Defended Prediction:**  \n{def_label} ({def_prob*100:.2f}%)  \n✅ Defense Successful")
        else:
            col_c.warning(f"**Defended Prediction:**  \n{def_label} ({def_prob*100:.2f}%)  \n⚠️ Defense Failed")

if __name__ == "__main__":
    main()
