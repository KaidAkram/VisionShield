# 🛡️ VisionShield AI

VisionShield AI is an interactive, portfolio-grade Red Teaming and Defense evaluation suite. It is designed to test Convolutional Neural Networks (specifically a pre-trained ResNet-18) against white-box adversarial perturbations and demonstrate defensive preprocessing mitigations to restore model integrity.

## 🧠 Project Overview

Modern deep learning systems are highly susceptible to **Adversarial Attacks**—subtle, mathematically calculated perturbations added to inputs that are imperceptible to the human eye but cause catastrophic misclassification by the AI. VisionShield AI exposes these vulnerabilities and provides an interactive playground to test defensive strategies.

### What We Built
- **Target Model Integration**: A fully mapped ResNet-18 pipeline locking parameters (`requires_grad=False`) for clean evaluation.
- **Offensive Pipeline (Red Teaming)**: Implementation of core mathematical exploits including the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD).
- **Defensive Engineering**: Preprocessing countermeasures including Gaussian Smoothing and Bit-Depth Reduction (Quantization) to strip out high-frequency adversarial noise.
- **Interactive UI**: A Streamlit dashboard to visually analyze the shifting confidence dynamics between clean inputs, adversarial noise, and defended outcomes.

---

## ⚔️ Understanding White-Box Attacks

In a **White-Box Attack**, the adversary has full access to the target model's architecture, weights, and gradients. By performing a forward pass and calculating the loss with respect to the input pixels (rather than the weights), the attacker can calculate exactly how to adjust the input image to maximize the model's error.

### How it Affects AI Systems Invisibly
The perturbation is constrained by an **Epsilon** ($\epsilon$) budget, meaning the pixel changes are strictly bounded. To a human observer, the image looks entirely normal and uncorrupted. However, the model perceives these micro-adjustments as overwhelming evidence for an incorrect class.

```mermaid
graph TD
    A[Clean Image (Jellyfish)] -->|Visually Identical| C{Human Observer}
    C -->|Perceives| D[Jellyfish]
    
    A -->|Calculated Gradient Addition| B[Adversarial Image (Epsilon bounded)]
    B -->|Visually Identical| C
    
    A -->|Forward Pass| E[ResNet-18]
    E -->|Confidence: 99%| F[Jellyfish]
    
    B -->|Forward Pass| G[ResNet-18]
    G -->|Confidence: 100%| H[Dishrag]
```

### The Mathematical Exploits
1. **Fast Gradient Sign Method (FGSM)**: A single-step attack that calculates the gradient of the loss and shifts the input pixels one step in the direction that maximizes that loss.
2. **Projected Gradient Descent (PGD)**: A more powerful iterative version of FGSM. It takes multiple smaller steps, clipping the perturbation back to the valid $\epsilon$-neighborhood at each iteration.

---

## 🛡️ How AI Engineers Should Prevent These Attacks

Relying purely on model architecture is often insufficient against white-box attacks. Engineers must adopt a **Defense-in-Depth** strategy:

1. **Adversarial Training**: The most robust defense involves generating adversarial examples during the training phase and explicitly teaching the model to classify them correctly. This smooths out the decision boundaries.
2. **Input Preprocessing (As used in VisionShield)**: 
   - **Spatial Smoothing**: Applying filters (like Gaussian Blur) averages out the high-frequency micro-perturbations injected by the attacker.
   - **Quantization (Bit-Depth Reduction)**: Reducing the precision of the pixel values breaks the continuous mathematical gradients attackers rely upon.
3. **Gradient Masking/Obfuscation**: Employing non-differentiable layers or stochasticity to make it mathematically impossible (or extremely difficult) for the attacker to calculate useful gradients. *Note: Advanced attackers can sometimes bypass this by training a substitute model.*
4. **Ensemble Defenses**: Using multiple models with different architectures. An attack optimized for one model's gradients may not successfully transfer to another.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- PyTorch & Torchvision
- Streamlit

### Installation
```bash
git clone <your-repository-url>
cd VisionShield
pip install -r requirements.txt
```

### Running the Dashboard
```bash
streamlit run app.py
```
Upload an image or let the system generate a synthetic sample, tune your attack budgets in the sidebar, and evaluate how the defensive layers mitigate the exploits!
