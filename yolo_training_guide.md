# YOLOv8 Terminal Training Guide

Because YOLOv8 training is highly computationally expensive, running it directly inside the Streamlit app blocks the user interface and doesn't easily allow for pausing and resuming. 

For longer training sessions (e.g. 50+ epochs), it is highly recommended to run the training in a separate terminal window. This allows you to pause the training at any time without losing your progress, and keeps your DeepSight app responsive.

---

## 1. Starting a New Training Session

To start training from scratch on the URPC2020 dataset, open a **new terminal window** in VS Code (Terminal > New Terminal), ensure you are in your project folder (`c:\Users\samee\OneDrive\Desktop\IDS Project`), and run:

```powershell
yolo detect train data=d:/IDS/URPC2020/data.yaml model=yolov8n.pt epochs=50 imgsz=640 project=runs/detect name=yolo_custom_train
```

> [!TIP]
> If you have a dedicated NVIDIA GPU, ultralytics will automatically detect and use it, which makes training 5-10x faster.

---

## 2. Pausing the Training (Graceful Stop)

If you need to stop the training mid-way (to free up CPU/GPU resources or turn off your computer):

1. Go to the terminal where the training is running.
2. Press **`Ctrl + C`** on your keyboard.
3. Wait a few seconds. YOLOv8 will catch the interrupt, finish saving the weights of the *last fully completed epoch*, and then shut down cleanly.

The weights from the exact moment you stopped are saved at:
`runs/detect/yolo_custom_train/weights/last.pt`

---

## 3. Resuming the Training

When you are ready to continue training, you **do not** start from scratch. You tell YOLO to pick up exactly where it left off using the `last.pt` file.

In your terminal, run:

```powershell
yolo detect train resume=True model=runs/detect/yolo_custom_train/weights/last.pt
```

YOLO will read the saved state, load the optimizer, and continue from the exact epoch it stopped at, running until it hits the original 50 epoch target.

---

## 4. Hooking the Model back into DeepSight

Once the training successfully finishes (whether in one go or after resuming multiple times), the final, best-performing model weights will be saved as `best.pt`.

To use this model in the **Step 7 Live Demo** of your app:

1. Locate the best weights file at: `runs/detect/yolo_custom_train/weights/best.pt`
2. Copy that `best.pt` file into your main project folder (`c:\Users\samee\OneDrive\Desktop\IDS Project`)
3. Rename the file to **`yolo_custom.pt`**
4. Refresh your Streamlit app. Step 7 will automatically detect `yolo_custom.pt` and use it instead of the generic land-object detector!
