"""
detection.py — Salient Object Detection Module
Uses morphological operations and contours to find the fish in enhanced images.
"""
import cv2
import numpy as np

def detect_salient_object(img: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """
    Detects the most salient object (the fish) using U-2-Net AI Background Removal.
    Returns:
        bbox: (x, y, w, h) of the bounding box.
        cropped: the perfectly masked, cropped image of the object.
    """
    try:
        from rembg import remove, new_session
        import PIL.Image
        
        # Initialize U-2-Net session (downloads weights on first run ~170MB)
        session = new_session('u2net')
        
        # Convert to PIL Image for rembg
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = PIL.Image.fromarray(img_rgb)
        
        # AI Background Removal (Returns RGBA)
        out_pil = remove(pil_img, session=session)
        out_np = np.array(out_pil)
        
        # Alpha channel is the perfect mask of the fish
        alpha = out_np[:, :, 3]
        
        y_indices, x_indices = np.where(alpha > 0)
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            h, w = img.shape[:2]
            return (0, 0, w, h), img.copy()
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Add padding margin
        margin = 15
        H, W = img.shape[:2]
        x1 = max(0, x_min - margin)
        y1 = max(0, y_min - margin)
        x2 = min(W, x_max + margin)
        y2 = min(H, y_max + margin)
        
        # Apply the perfect mask to the original BGR image
        mask = (alpha > 0).astype(np.uint8)
        img_masked = img * mask[:, :, np.newaxis]
        
        cropped = img_masked[y1:y2, x1:x2].copy()
        
        return (x1, y1, x2 - x1, y2 - y1), cropped
        
    except Exception as e:
        print(f"Rembg detection failed: {e}")
        h, w = img.shape[:2]
        return (0, 0, w, h), img.copy()

def detect_from_mask(img: np.ndarray, mask_path: str) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """
    Uses an existing ground truth mask to perfectly crop the fish.
    Returns:
        bbox: (x, y, w, h) of the bounding box.
        cropped: the perfectly masked, cropped image of the object.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return detect_salient_object(img)
        
    # Resize mask to match image size if necessary
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    y_indices, x_indices = np.where(mask > 0)
    
    if len(y_indices) == 0 or len(x_indices) == 0:
        h, w = img.shape[:2]
        return (0, 0, w, h), img.copy()
        
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    
    # Add padding margin
    margin = 15
    H, W = img.shape[:2]
    x1 = max(0, x_min - margin)
    y1 = max(0, y_min - margin)
    x2 = min(W, x_max + margin)
    y2 = min(H, y_max + margin)
    
    # Apply mask
    binary_mask = (mask > 0).astype(np.uint8)
    img_masked = img * binary_mask[:, :, np.newaxis]
    
    cropped = img_masked[y1:y2, x1:x2].copy()
    
    return (x1, y1, x2 - x1, y2 - y1), cropped

def draw_bounding_box(img: np.ndarray, bbox: tuple[int, int, int, int], color=(0, 255, 0), thickness=3) -> np.ndarray:
    """
    Draws a bounding box on the image.
    """
    res = img.copy()
    x, y, w, h = bbox
    cv2.rectangle(res, (x, y), (x + w, y + h), color, thickness)
    
    # Add a label background and text for "Object"
    label = "Fish detected"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    
    cv2.rectangle(res, (x, y - text_size[1] - 10), (x + text_size[0] + 10, y), color, -1)
    cv2.putText(res, label, (x + 5, y - 5), font, font_scale, (0, 0, 0), font_thickness)
    
    return res
