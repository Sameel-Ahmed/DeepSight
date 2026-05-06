"""
enhancement.py — White Balance Correction + CLAHE pipeline.
Matches the specification from Deliverable 3.
"""
import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as _psnr


# ── Core functions ────────────────────────────────────────────────────────────

def white_balance(img: np.ndarray) -> np.ndarray:
    """
    Grey World Assumption white balance in LAB colour space.
    Scales the A and B channels to remove the blue-green cast
    typical of underwater imagery.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    avg_a = np.mean(lab[:, :, 1])
    avg_b = np.mean(lab[:, :, 2])
    lab[:, :, 1] -= (avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1
    lab[:, :, 2] -= (avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_clahe(img: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE on the L channel of LAB colour space.
    Improves local contrast while preserving hue (A, B channels untouched).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def enhance_image(img: np.ndarray) -> np.ndarray:
    """Full pipeline: White Balance → CLAHE."""
    return apply_clahe(white_balance(img))


def compute_psnr(img_ref: np.ndarray, img_enh: np.ndarray) -> float:
    """PSNR between reference and enhanced image (higher = better)."""
    ref = cv2.resize(img_ref, (256, 256))
    enh = cv2.resize(img_enh, (256, 256))
    return float(_psnr(ref, enh, data_range=255))


# ── Batch processing ──────────────────────────────────────────────────────────

def enhance_batch(image_paths: list,
                  output_folder: str,
                  reference_paths: list | None = None,
                  progress_cb=None) -> list:
    """
    Enhance all images and optionally compute PSNR against references.

    Returns list of dicts:
        path, enhanced_path, raw (ndarray), enhanced (ndarray), psnr (float|None)
    """
    os.makedirs(output_folder, exist_ok=True)

    # Build reference lookup by filename
    ref_map: dict = {}
    if reference_paths:
        for rp in reference_paths:
            ref_map[os.path.basename(rp)] = rp

    results = []
    n = len(image_paths)

    for i, path in enumerate(image_paths):
        raw = cv2.imread(path)
        if raw is None:
            if progress_cb:
                progress_cb((i + 1) / n)
            continue

        raw_r   = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_AREA)
        enhanced = enhance_image(raw_r)

        out_path = os.path.join(output_folder, os.path.basename(path))
        cv2.imwrite(out_path, enhanced)

        psnr_val = None
        fname = os.path.basename(path)
        if fname in ref_map:
            ref = cv2.imread(ref_map[fname])
            if ref is not None:
                psnr_val = compute_psnr(ref, enhanced)

        results.append({
            'path':          path,
            'enhanced_path': out_path,
            'raw':           raw_r,
            'enhanced':      enhanced,
            'psnr':          psnr_val,
        })

        if progress_cb:
            progress_cb((i + 1) / n)

    return results
