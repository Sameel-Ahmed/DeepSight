# AquaVision Pipeline — Setup & Usage Guide

## Prerequisites
- Python 3.10 or higher
- pip
- ~2 GB free disk space for datasets

---

## Step 1 — Get the Project Files

Copy all project files into one folder, for example:
```
C:\IDS-Project\
```

The structure should look like:
```
IDS-Project/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── pipeline/
│   ├── __init__.py
│   ├── ingestion.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── enhancement.py
│   ├── features.py
│   └── model.py
└── data/
    └── enhanced/        ← created automatically
```

---

## Step 2 — Install Dependencies

Open a terminal (Command Prompt or PowerShell on Windows, Terminal on Mac/Linux)
and navigate to the project folder:

```bash
cd C:\IDS-Project
```

Install all required packages:

```bash
pip install -r requirements.txt
```

This installs: streamlit, opencv-python, numpy, pandas, scikit-learn,
scikit-image, plotly, joblib, Pillow, matplotlib, seaborn.

---

## Step 3 — Download Datasets from Kaggle

You need a free Kaggle account. Go to https://www.kaggle.com/settings and
generate an API token (downloads kaggle.json).

### Dataset A — UIEB (for Enhancement + PSNR)
URL: https://www.kaggle.com/datasets/larjeck/uieb-dataset-raw

Download and extract. You should have a folder like:
```
uieb-dataset-raw/
├── raw-890/          ← raw underwater images
└── reference890/     ← matching high-quality references
```

Rename the subfolders if needed so they match one of these names:
- Raw folder    : raw, raw-890, raw_890, raw_images, input
- Reference folder: reference, reference890, ref, gt, ground_truth

### Dataset B — Large Scale Fish Dataset (for Classification)
URL: https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset

Download and extract. It should have a structure like:
```
a-large-scale-fish-dataset/
├── Fish_Dataset/
│   ├── Black Sea Sprat/
│   ├── Gilt-Head Bream/
│   ├── Horse Mackerel/
│   └── ... (9 classes total)
```

Point the app at the Fish_Dataset/ subfolder, not the outer folder.

---

## Step 4 — Run the App

In your terminal (inside the project folder):

```bash
streamlit run app.py
```

Your browser will open automatically at:
```
http://localhost:8501
```

If it doesn't open, paste that URL into your browser manually.

---

## Step 5 — Using the App (Follow the Pipeline)

Navigate using the sidebar. Complete each step in order:

### 1. Data Ingestion
- Paste the full path to your dataset folder in the text box
- Example: C:\datasets\uieb-dataset-raw
- Click "Load Dataset" — the app will auto-detect the structure
- You'll see total images, references, and classes detected

### 2. Preprocessing
- Click "Run Preprocessing"
- See sample images resized to 256×256
- Check the valid/corrupted image count

### 3. EDA
- Click "Run EDA"
- Explore the Channel Analysis tab — confirms blue-green dominance
- Check Class Distribution if using the fish dataset
- Read the 3 auto-generated insights

### 4. Enhancement
- Use the slider to choose how many images to enhance
- Click "Run Enhancement"
- View before/after comparisons
- If using UIEB with references, PSNR scores are shown (target: 18–24 dB)

### 5. Feature Extraction  (requires Fish Dataset loaded)
- Adjust the slider for how many images to process
- Click "Extract Features"
- See the 70-feature matrix stats and distribution plots

### 6. Model Training  (requires Step 5 complete)
- Click "Train Model"
- View Accuracy, F1-Score, Confusion Matrix, Per-class Metrics
- Model is saved to model.pkl automatically

### 7. Live Demo
- Upload any underwater image (jpg, png)
- See: Original | Enhanced | PSNR | RGB Histograms | Species Prediction
- Works with any underwater image, not just dataset images

---

## Recommended Workflow for the Project Demo

Run the full pipeline in this order:

1. Load the UIEB dataset → complete steps 1–4 to show enhancement quality
2. Load the Fish dataset → complete steps 1, 2, 5, 6 to show classification
3. Use the Live Demo with a few sample images from both datasets

For the viva:
- Navigate through each sidebar page to show each pipeline step
- Highlight the 3 EDA insights and explain the channel imbalance
- Show before/after images in Enhancement
- Show the confusion matrix and F1-score in Model Training
- Upload a test image in Live Demo for the live prediction

---

## GitHub Upload

```bash
git init
git add .
git commit -m "IDS Semester Project - Underwater Image Pipeline"
git remote add origin https://github.com/YOUR_USERNAME/underwater-pipeline.git
git push -u origin main
```

Add a .gitignore to exclude datasets and model:
```
data/
model.pkl
__pycache__/
*.pyc
.streamlit/secrets.toml
```

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'cv2'"**
→ Run: pip install opencv-python

**"streamlit: command not found"**
→ Run: python -m streamlit run app.py

**App shows blank page**
→ Hard refresh: Ctrl+Shift+R

**PSNR shows N/A**
→ Your dataset doesn't have paired references, or filenames don't match between
  raw/ and reference/ folders. Enhancement still works, just no PSNR.

**Classification tab is locked**
→ The dataset must be in Classification mode (class subfolders). Load the Fish
  Dataset for Steps 5 and 6.

**Very slow feature extraction**
→ Reduce the slider value. 500–1000 images is enough for a good model.
