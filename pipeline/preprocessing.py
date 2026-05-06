"""
preprocessing.py — Image loading, resizing, validation.
"""
import cv2
import numpy as np

TARGET_SIZE = (256, 256)


def load_image(path: str, size: tuple = TARGET_SIZE) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def load_sample(paths: list, size: tuple = TARGET_SIZE,
                max_n: int = 60) -> tuple[list, list]:
    """
    Load up to max_n images for display purposes.
    Returns (loaded_list, failed_list)
        loaded_list : list of (path, img_bgr)
    """
    loaded, failed = [], []
    for p in paths[:max_n]:
        img = load_image(p, size)
        if img is not None:
            loaded.append((p, img))
        else:
            failed.append(p)
    return loaded, failed


def get_dataset_stats(paths: list, probe: int = 200) -> dict:
    """
    Compute size / health stats by probing up to `probe` images.
    """
    sample = paths[:probe]
    heights, widths, valid, invalid = [], [], 0, 0

    for p in sample:
        img = cv2.imread(p)
        if img is None:
            invalid += 1
            continue
        h, w = img.shape[:2]
        heights.append(h)
        widths.append(w)
        valid += 1

    if not heights:
        return {'valid': 0, 'invalid': invalid,
                'total': len(paths), 'probe': len(sample)}

    return {
        'valid':   valid,
        'invalid': invalid,
        'total':   len(paths),
        'probe':   len(sample),
        'min_res': f"{min(heights)}×{min(widths)}",
        'max_res': f"{max(heights)}×{max(widths)}",
        'avg_res': f"{int(np.mean(heights))}×{int(np.mean(widths))}",
    }
