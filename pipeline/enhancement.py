"""
enhancement.py — Configurable 7-Stage Underwater Enhancement Pipeline.
Each stage can be toggled on/off independently via active_stages parameter.
"""
import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio as _psnr


# ── All available stage keys (ordered) ───────────────────────────────────────
STAGE_KEYS = [
    'red_compensation',
    'white_balance',
    'gamma',
    'clahe',
    'unsharp',
    'bilateral',
    'histogram_stretch',
]

# ── Stage metadata for UI rendering ──────────────────────────────────────────
STAGE_META = {
    'red_compensation': {
        'label': 'Stage 1 · Red Channel Compensation',
        'desc':  'Corrects the red/orange colour cast caused by underwater light attenuation by boosting the red channel toward the green channel mean.',
        'icon':  '🔴',
    },
    'white_balance': {
        'label': 'Stage 2 · LAB White Balance',
        'desc':  'Removes the dominant blue-green tint using the grey-world assumption in LAB colour space for a neutral, natural look.',
        'icon':  '⚖️',
    },
    'gamma': {
        'label': 'Stage 3 · Gamma Correction',
        'desc':  'Brightens dark shadow regions and recovers hidden detail without overexposing highlights, using a power-law transform.',
        'icon':  '☀️',
    },
    'clahe': {
        'label': 'Stage 4 · CLAHE',
        'desc':  'Contrast Limited Adaptive Histogram Equalization — equalises local contrast region-by-region to avoid the washed-out look of global equalization.',
        'icon':  '📊',
    },
    'unsharp': {
        'label': 'Stage 5 · Unsharp Mask',
        'desc':  'Enhances fine surface textures (fish scales, fins) by subtracting a Gaussian-blurred copy from the original, amplifying edge detail.',
        'icon':  '🔬',
    },
    'bilateral': {
        'label': 'Stage 6 · Bilateral Denoising',
        'desc':  'Removes sensor grain and compression noise while keeping sharp edges fully intact — an edge-aware alternative to Gaussian blur.',
        'icon':  '🌊',
    },
    'histogram_stretch': {
        'label': 'Stage 7 · Histogram Stretching',
        'desc':  'Stretches the per-channel pixel range to the full 0–255 spectrum, recovering vivid colours from washed-out or low-contrast images.',
        'icon':  '🎨',
    },
}


# ── Core Stage Functions ──────────────────────────────────────────────────────

def red_channel_compensation(img: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Stage 1: Compensates the red channel using the green channel."""
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)
    mean_g, mean_r = np.mean(g), np.mean(r)
    if mean_r < mean_g:
        r = r + alpha * (mean_g - mean_r) * (1 - r / 255.0)
    r = np.clip(r, 0, 255)
    return cv2.merge((b, g, r)).astype(np.uint8)


def white_balance(img: np.ndarray) -> np.ndarray:
    """Stage 2: Grey World white balance in LAB colour space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float64)
    avg_a = np.mean(lab[:, :, 1])
    avg_b = np.mean(lab[:, :, 2])
    lab[:, :, 1] -= (avg_a - 128) * (lab[:, :, 0] / 255.0) * 1.1
    lab[:, :, 2] -= (avg_b - 128) * (lab[:, :, 0] / 255.0) * 1.1
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma_correction(img: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Stage 3: Adjusts brightness via power-law transform."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Stage 4: CLAHE on the L channel of LAB colour space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def unsharp_mask(img: np.ndarray, kernel_size: tuple = (5, 5), sigma: float = 1.0, amount: float = 1.5) -> np.ndarray:
    """Stage 5: Enhances fine textures via unsharp masking."""
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img.astype(np.float32) - float(amount) * blurred.astype(np.float32)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def bilateral_denoise(img: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
    """Stage 6: Edge-aware noise reduction using Bilateral Filter."""
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def histogram_stretch(img: np.ndarray) -> np.ndarray:
    """Stage 7: Per-channel min-max histogram stretching to full 0-255 range."""
    result = np.zeros_like(img, dtype=np.float32)
    for c in range(3):
        ch = img[:, :, c].astype(np.float32)
        lo, hi = ch.min(), ch.max()
        if hi > lo:
            result[:, :, c] = (ch - lo) / (hi - lo) * 255.0
        else:
            result[:, :, c] = ch
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Map key -> function ───────────────────────────────────────────────────────
_STAGE_FN = {
    'red_compensation':  red_channel_compensation,
    'white_balance':     white_balance,
    'gamma':             gamma_correction,
    'clahe':             apply_clahe,
    'unsharp':           unsharp_mask,
    'bilateral':         bilateral_denoise,
    'histogram_stretch': histogram_stretch,
}


# ── Public API ────────────────────────────────────────────────────────────────

def enhance_image(img: np.ndarray, active_stages: set | None = None) -> np.ndarray:
    """
    Apply the selected enhancement stages in order.
    If active_stages is None, all 7 stages are applied (original default behaviour).
    """
    if active_stages is None:
        active_stages = set(STAGE_KEYS)
    for key in STAGE_KEYS:
        if key in active_stages:
            img = _STAGE_FN[key](img)
    return img


def enhance_image_stages(img: np.ndarray, active_stages: set | None = None) -> tuple:
    """
    Run each active stage individually and capture before/after for each.

    Returns:
        stages_out : list of dicts with keys:
            key, label, desc, icon, before (ndarray), after (ndarray)
        final_img  : the fully enhanced image
    """
    if active_stages is None:
        active_stages = set(STAGE_KEYS)

    stages_out = []
    current = img.copy()

    for key in STAGE_KEYS:
        if key in active_stages:
            meta   = STAGE_META[key]
            before = current.copy()
            current = _STAGE_FN[key](current)
            stages_out.append({
                'key':    key,
                'label':  meta['label'],
                'desc':   meta['desc'],
                'icon':   meta['icon'],
                'before': before,
                'after':  current.copy(),
            })

    return stages_out, current


def compute_psnr(img_ref: np.ndarray, img_enh: np.ndarray) -> float:
    """PSNR between reference and enhanced image (higher = better)."""
    ref = cv2.resize(img_ref, (256, 256))
    enh = cv2.resize(img_enh, (256, 256))
    return float(_psnr(ref, enh, data_range=255))


# ── Batch processing ──────────────────────────────────────────────────────────

def enhance_batch(image_paths: list,
                  output_folder: str,
                  reference_paths: list | None = None,
                  active_stages: set | None = None,
                  progress_cb=None) -> list:
    """
    Enhance all images using the selected stages.

    Returns list of dicts:
        path, enhanced_path, raw (ndarray), enhanced (ndarray), psnr (float|None)
    """
    os.makedirs(output_folder, exist_ok=True)

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

        raw_r    = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_AREA)
        enhanced = enhance_image(raw_r, active_stages)

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
