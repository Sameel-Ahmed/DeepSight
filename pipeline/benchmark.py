"""
benchmark.py — Runs standard academic enhancement benchmarks (PSNR & SSIM) 
on paired datasets like UIEB.
"""
import os
import cv2
import numpy as np
from pathlib import Path
from pipeline.enhancement import enhance_image, compute_psnr, compute_ssim

def run_uieb_benchmark(images: list, references: list, active_stages: set = None, progress_cb=None) -> dict:
    """
    Computes average PSNR and SSIM for all images that have a corresponding reference.
    Matches raw and reference images by base filename.
    """
    if not images or not references:
        return {'mean_psnr': 0.0, 'mean_ssim': 0.0, 'total': 0, 'results': []}

    # Map references by their exact filename to handle ordering issues
    ref_map = {Path(rp).name: rp for rp in references}

    total = len(images)
    results = []
    
    psnr_scores = []
    ssim_scores = []

    for i, raw_path in enumerate(images):
        fname = Path(raw_path).name
        if fname not in ref_map:
            if progress_cb: progress_cb((i + 1) / total)
            continue
            
        raw = cv2.imread(raw_path)
        ref = cv2.imread(ref_map[fname])
        
        if raw is None or ref is None:
            if progress_cb: progress_cb((i + 1) / total)
            continue
            
        # Standardise sizes for speed
        raw_r = cv2.resize(raw, (256, 256), interpolation=cv2.INTER_AREA)
        
        # Apply the currently selected enhancement pipeline
        try:
            enhanced = enhance_image(raw_r, active_stages)
            
            p = compute_psnr(ref, enhanced)
            s = compute_ssim(ref, enhanced)
            
            psnr_scores.append(p)
            ssim_scores.append(s)
            
            results.append({
                'filename': fname,
                'psnr': p,
                'ssim': s
            })
        except Exception as e:
            print(f"Error enhancing {fname}: {e}")
            
        if progress_cb:
            progress_cb((i + 1) / total)

    mean_psnr = float(np.mean(psnr_scores)) if psnr_scores else 0.0
    mean_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0

    return {
        'mean_psnr': mean_psnr,
        'mean_ssim': mean_ssim,
        'total_evaluated': len(psnr_scores),
        'results': results
    }
