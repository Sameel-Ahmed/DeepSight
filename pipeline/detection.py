"""
detection.py — Salient Object Detection Module
Uses morphological operations and contours to find the fish in enhanced images.
"""
import cv2
import numpy as np

def detect_salient_object(img: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """
    Detects the most salient object (likely the fish) in an enhanced underwater image.
    Returns:
        bbox: (x, y, w, h) of the bounding box.
        cropped: the cropped image of the object.
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # 3. Morphological gradient to highlight edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
    
    # 4. Thresholding to create a binary mask (Otsu's method)
    _, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 5. Morphological closing to fill gaps
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    
    # 6. Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback: return the whole image if nothing is found
        h, w = img.shape[:2]
        return (0, 0, w, h), img.copy()
    
    # 7. Assume the largest contour is the object of interest (the fish)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Filter out tiny contours just in case
    if cv2.contourArea(largest_contour) < 500:
        h, w = img.shape[:2]
        return (0, 0, w, h), img.copy()
        
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Add a small margin (padding) to the bounding box
    margin = 15
    H, W = img.shape[:2]
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(W, x + w + margin)
    y2 = min(H, y + h + margin)
    
    # Add GrabCut for pixel-perfect foreground extraction
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    rect = (x1, y1, x2 - x1, y2 - y1)
    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        img_masked = img * mask2[:, :, np.newaxis]
    except Exception:
        img_masked = img
        
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
