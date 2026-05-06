"""
features.py — Handcrafted feature extraction for classification.
70 features per image: RGB stats (6) + colour histograms (48) + LBP texture (16).
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


FEATURE_DIM = 70  # 6 + 48 + 16


def extract_features(img: np.ndarray) -> list:
    img = cv2.resize(img, (256, 256))

    feats = []

    # 1. Per-channel mean & std  (6 features)
    for c in range(3):
        feats.append(float(np.mean(img[:, :, c])))
        feats.append(float(np.std(img[:, :, c])))

    # 2. Normalised colour histograms – 16 bins × 3 channels  (48 features)
    for c in range(3):
        hist = cv2.calcHist([img], [c], None, [16], [0, 256]).flatten()
        hist /= (hist.sum() + 1e-7)
        feats.extend(hist.tolist())

    # 3. LBP texture histogram – 16 bins  (16 features)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp  = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=16, range=(0, 16))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-7)
    feats.extend(lbp_hist.tolist())

    return feats


def feature_names() -> list:
    names = []
    ch = ['B', 'G', 'R']
    for c in ch:
        names += [f'{c}_mean', f'{c}_std']
    for c in ch:
        names += [f'{c}_hist_{i}' for i in range(16)]
    names += [f'LBP_{i}' for i in range(16)]
    return names


def build_feature_matrix(image_paths: list,
                          labels: list | None = None,
                          progress_cb=None) -> tuple:
    """
    Returns (X: ndarray, y: ndarray | None)
    Skips unreadable images; corresponding labels are also skipped.
    """
    X, y = [], []
    n = len(image_paths)

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            if progress_cb:
                progress_cb((i + 1) / n)
            continue
        X.append(extract_features(img))
        if labels is not None and i < len(labels):
            y.append(labels[i])

        if progress_cb:
            progress_cb((i + 1) / n)

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32) if labels is not None else None
    return X_arr, y_arr
