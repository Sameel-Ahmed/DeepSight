# 🔬 DeepSight: Underwater Image Enhancement & Marine Classification

DeepSight is a complete data science pipeline designed to solve the physical and mathematical challenges of underwater photography. It restores color-attenuated images and classifies marine species using a hybrid of classical Machine Learning and modern Deep Learning.

## 🚀 Key Features
- **6-Stage Enhancement Pipeline:** Mathematically restores red light and removes haze using CLAHE, White Balance, and Gamma Correction.
- **Academic Benchmarking:** Native support for the UIEB dataset to compute PSNR and SSIM scores.
- **Hybrid AI Detection:** 
  - **YOLO11:** Real-time multi-object detection with bounding boxes.
  - **Random Forest/SVM:** Transparent species classification using 70-dimensional mathematical feature vectors (HOG, Color, LBP).
- **Cross-Dataset Validation:** Built-in tools to evaluate model generalization on the QUT Fish dataset.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/DeepSight.git
   cd DeepSight
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📁 Pipeline Structure
- `app.py`: Main Streamlit UI and orchestration.
- `pipeline/`: Core logic modules.
  - `enhancement.py`: The 6-stage restoration engine.
  - `benchmark.py`: PSNR/SSIM evaluation logic.
  - `features.py`: Mathematical feature extraction (HOG, LBP, RGB).
  - `model.py`: Classifier training and evaluation.
  - `ingestion.py`: Dataset parsing for Fish, UIEB, and QUT formats.

## 📊 Methodology
This project follows a two-phase approach:
1. **The Physics Phase:** Solving light attenuation using color channel compensation.
2. **The Data Phase:** Using extracted textures and shapes to identify species in "in-the-wild" murky conditions.

---
*Developed for Intro to Data Science · Spring 2026*
