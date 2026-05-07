"""
yolo_trainer.py — Fine-tune YOLOv8 on custom underwater datasets (URPC2020).
"""
import os
import shutil
from ultralytics import YOLO

def train_yolo(data_yaml: str, epochs: int = 50, imgsz: int = 640, output_model_name: str = "yolo_custom.pt"):
    """
    Fine-tunes YOLOv8n on the provided data.yaml.
    Copies the best resulting model to the project root.
    """
    # Load a pretrained YOLOv8 nano model
    model = YOLO('yolov8n.pt')
    
    # Train the model
    # Results are saved to runs/detect/train/ by default by ultralytics
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project='runs/detect',
        name='yolo_custom_train',
        exist_ok=True, # Overwrite previous run of same name
        device='' # Auto-detect GPU/CPU
    )
    
    # Best weights path
    best_weights_path = os.path.join('runs', 'detect', 'yolo_custom_train', 'weights', 'best.pt')
    
    # Copy best model to project root
    target_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), output_model_name)
    if os.path.exists(best_weights_path):
        shutil.copy(best_weights_path, target_path)
        return target_path
    
    return None
