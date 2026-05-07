"""
ingestion.py — Flexible dataset loader.
Supports: UIEB-style (raw + reference), classification (class subfolders), flat folder.
"""
import os
from pathlib import Path

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

# Aliases for raw/reference folder names across different dataset releases
RAW_ALIASES    = {'raw', 'raw-890', 'raw_890', 'raw_images', 'input',
                  'original', 'underwater', 'images', 'raws'}
REF_ALIASES    = {'reference', 'reference890', 'ref', 'gt', 'ground_truth',
                  'high_quality', 'hq', 'enhanced_gt', 'target'}


def _img_files(folder: Path) -> list:
    return sorted([str(p) for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS])


def _detect_mode(root: Path):
    subdirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}

    raw_dir = next((subdirs[k] for k in RAW_ALIASES if k in subdirs), None)
    ref_dir = next((subdirs[k] for k in REF_ALIASES if k in subdirs), None)

    if raw_dir and _img_files(raw_dir):
        return 'uieb', raw_dir, ref_dir

    # Check if every subdir is an image class folder
    image_subdirs = [d for d in subdirs.values() if _img_files(d)]
    if len(image_subdirs) >= 2:
        return 'classification', None, None

    # Check root itself for images
    root_imgs = _img_files(root)
    if root_imgs:
        return 'flat', None, None

    raise ValueError(
        f"Could not detect dataset structure in: {root}\n"
        "Expected one of:\n"
        "  • UIEB   – has a 'raw' subfolder (+ optional 'reference')\n"
        "  • Class  – has ≥2 subfolders each containing images\n"
        "  • Flat   – images directly in the folder"
    )


def load_dataset(root_path: str) -> dict:
    """
    Returns
    -------
    dict with keys:
        mode        : 'uieb' | 'classification' | 'flat'
        root        : absolute path string
        images      : list of raw image paths
        references  : list of reference image paths (uieb only, else [])
        labels      : list of int label indices (classification only, else [])
        class_map   : {idx: class_name}  (classification only, else {})
        class_names : list of class name strings
        total       : total image count
    """
    root = Path(root_path).resolve()
    if not root.exists():
        raise ValueError(f"Path does not exist: {root_path}")

    mode, raw_dir, ref_dir = _detect_mode(root)

    result = {
        'mode':        mode,
        'root':        str(root),
        'images':      [],
        'references':  [],
        'labels':      [],
        'class_map':   {},
        'class_names': [],
        'total':       0,
    }

    if mode == 'uieb':
        result['images'] = _img_files(raw_dir)
        if ref_dir and ref_dir.exists():
            result['references'] = _img_files(ref_dir)

    elif mode == 'classification':
        subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
        result['gt_paths'] = {}  # image_path -> gt_mask_path
        
        for idx, d in enumerate(subdirs):
            # Check if this class folder has nested folders (e.g. 'Trout' and 'Trout GT')
            class_subdirs = [sd for sd in d.iterdir() if sd.is_dir()]
            
            raw_imgs = []
            gt_imgs = []
            
            if class_subdirs:
                # E.g. Fish_Dataset structure
                for sd in class_subdirs:
                    if 'gt' in sd.name.lower() or 'ground_truth' in sd.name.lower() or 'mask' in sd.name.lower():
                        gt_imgs.extend(_img_files(sd))
                    else:
                        raw_imgs.extend(_img_files(sd))
            else:
                # Flat class structure (images directly in 'Trout/')
                raw_imgs.extend(_img_files(d))
            
            if not raw_imgs:
                continue
                
            # Pair GT masks by filename if available
            if gt_imgs:
                gt_map = {Path(p).stem: p for p in gt_imgs}
                for rp in raw_imgs:
                    stem = Path(rp).stem
                    if stem in gt_map:
                        result['gt_paths'][rp] = gt_map[stem]
                        
            result['images'].extend(raw_imgs)
            result['labels'].extend([idx] * len(raw_imgs))
            result['class_map'][idx] = d.name
            result['class_names'].append(d.name)

    else:  # flat
        result['images'] = _img_files(root)

    result['total'] = len(result['images'])
    return result
