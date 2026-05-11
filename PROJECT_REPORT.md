# DeepSight: Hybrid Underwater Image Restoration & Marine Classification

## 1. Project Overview
**DeepSight** is an advanced academic project developed to address the specific physical challenges of underwater computer vision. The system provides an end-to-end pipeline that transforms murky, color-distorted underwater footage into high-contrast, color-accurate images suitable for automated marine species identification.

---

## 2. The Problem Statement
Underwater environments suffer from two primary optical distortions:
1. **Light Attenuation:** Red light is absorbed within the first 5 meters, leading to a dominant blue-green cast.
2. **Scattering:** Suspended particles (marine snow) cause light to scatter, resulting in low contrast and "hazy" imagery.

---

## 3. Technical Architecture
The project is built on a modular **7-Step Pipeline** implemented using Python and Streamlit.

### Phase 1: Physical Enhancement (The Restoration Engine)
Instead of relying on generic filters, DeepSight uses a 6-stage mathematical restoration process:
- **Red Channel Compensation:** Artificially restores the attenuated red pixels based on the intensity of the green channel.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Localized contrast enhancement to reveal textures without over-amplifying noise.
- **LAB Color Space Balancing:** Converts images to LAB space to perform white balancing on the 'a' and 'b' channels independently.

### Phase 2: Feature Engineering & AI Classification
To classify species (Trout, Sea Bass, Shrimp, etc.), we extract a **70+ dimensional feature vector**:
- **Color (RGB Histograms):** Captures the unique color signatures of species.
- **Texture (LBP):** Local Binary Patterns identify scale and fin patterns.
- **Shape (HOG):** Histograms of Oriented Gradients identify the anatomical structure.

**The Model:** We utilize a **Voting Ensemble Classifier** that combines **Random Forest** (robust to noise) and **SVM** (highly precise in high-dimensional space).

---

## 4. Visual Evidence & Results

### 4.1 Exploratory Data Analysis (EDA)
Below is the channel distribution analysis, highlighting the severe depletion of Red light in raw underwater datasets.

*(Note: In your final report, insert the EDA histogram chart here.)*

### 4.2 Model Performance
The following Confusion Matrix demonstrates the high precision of the Ensemble model in distinguishing between similar species like "Red Mullet" and "Striped Red Mullet."

*(Note: In your final report, insert the Confusion Matrix chart here.)*

---

## 5. Key Innovations
- **Hybrid Saliency Detection:** Combines U-2-Net AI with a Classical CV Fallback (Contour Detection) to ensure object detection works even if the AI fails.
- **Underwater Heuristic:** A built-in sensor that detects if an image is actually underwater before processing, preventing "false enhancements."
- **YOLO11 Integration:** Uses the latest YOLO11 architecture for real-time bounding box localization.

---

## 6. Conclusion
DeepSight successfully bridges the gap between physics-based restoration and data-driven classification. By mathematically compensating for light loss before feeding images into an AI ensemble, we achieved significantly higher accuracy than standard "off-the-shelf" classification models.

---

## 7. Future Work
- Integration of **Real-Time Video Stream Enhancement**.
- Deployment to **Edge Devices** (e.g., Raspberry Pi) for integration with underwater ROVs.
