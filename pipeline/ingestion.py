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


def _has_images_deep(folder: Path) -> bool:
    """Recursively checks if a folder or any of its subfolders contain images."""
    try:
        if _img_files(folder):
            return True
        for sd in folder.iterdir():
            if sd.is_dir() and _has_images_deep(sd):
                return True
    except PermissionError:
        pass
    return False


def _detect_mode(root: Path):
    try:
        subdirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    except (PermissionError, FileNotFoundError):
        subdirs = {}

    # Check for UIEB format
    raw_dir = next((subdirs[k] for k in RAW_ALIASES if k in subdirs), None)
    ref_dir = next((subdirs[k] for k in REF_ALIASES if k in subdirs), None)
    if raw_dir and _img_files(raw_dir):
        return 'uieb', raw_dir, ref_dir

    # Check for QUT format
    if (root / 'final_all_index.txt').exists() and (root / 'images' / 'raw_images').exists():
        return 'qut', root / 'images' / 'raw_images', root / 'final_all_index.txt'

    # Check for Classification format (Root contains classes)
    # A folder is a dataset if it has >= 2 subdirs that deeply contain images (ignoring GT/mask folders)
    valid_classes = [d for d in root.iterdir() if d.is_dir() and _has_images_deep(d) and not any(k in d.name.lower() for k in ['gt', 'mask', 'ground_truth'])]
    if len(valid_classes) >= 2:
        return 'classification', None, None
        
    # Check if any subdir is actually the dataset root (one level deep)
    for sd in root.iterdir():
        if not sd.is_dir(): continue
        valid_nested = [d for d in sd.iterdir() if d.is_dir() and _has_images_deep(d) and not any(k in d.name.lower() for k in ['gt', 'mask', 'ground_truth'])]
        if len(valid_nested) >= 2:
            return 'classification', sd, None

    # Check root itself for images (Flat)
    if _img_files(root):
        return 'flat', None, None

    raise ValueError(
        f"Could not detect dataset structure in: {root}\n"
        "Expected one of:\n"
        "  • UIEB   – has a 'raw' subfolder (+ optional 'reference')\n"
        "  • QUT    – has 'images/raw_images' and 'final_all_index.txt'\n"
        "  • Class  – has ≥2 subfolders each containing images\n"
        "  • Flat   – images directly in the folder"
    )


def load_dataset(root_path: str) -> dict:
    # Clean the path (remove quotes if user copy-pasted with quotes)
    root_path = root_path.strip().strip('"').strip("'")
    
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

    elif mode == 'qut':
        index_file = ref_dir
        with open(index_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        unique_classes = {}
        for line in lines:
            parts = line.strip().split('=')
            if len(parts) >= 4:
                try:
                    class_id = int(parts[0]) - 1  # 0-indexed for training
                    class_name = parts[1]
                    filename = parts[3] + '.jpg'
                    
                    img_path = str(raw_dir / filename)
                    if os.path.exists(img_path):
                        result['images'].append(img_path)
                        result['labels'].append(class_id)
                        unique_classes[class_id] = class_name
                except ValueError:
                    continue
                    
        max_id = max(unique_classes.keys()) if unique_classes else -1
        result['class_names'] = [unique_classes.get(i, f"Class_{i}") for i in range(max_id + 1)]
        result['class_map'] = unique_classes

    elif mode == 'classification':
        base_dir = raw_dir if raw_dir else root
        subdirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
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
                
            # Pair GT masks by filename (Fuzzy Matching)
            if gt_imgs:
                gt_map = {Path(p).stem.lower(): p for p in gt_imgs}
                for rp in raw_imgs:
                    r_stem = Path(rp).stem.lower()
                    # 1. Try exact match
                    if r_stem in gt_map:
                        result['gt_paths'][rp] = gt_map[r_stem]
                        continue
                    
                    # 2. Try fuzzy match (e.g. '00001' matching '00001_gt' or '00001_mask')
                    for g_stem, g_path in gt_map.items():
                        if r_stem in g_stem or g_stem in r_stem:
                            result['gt_paths'][rp] = g_path
                            break
                        
            result['images'].extend(raw_imgs)
            result['labels'].extend([idx] * len(raw_imgs))
            result['class_map'][idx] = d.name
            result['class_names'].append(d.name)

    else:  # flat
        result['images'] = _img_files(root)

    result['total'] = len(result['images'])
    return result
