# Training a custom object detector

The live pipeline uses a YOLOv8 model for in-cabin objects. A **stock COCO**
model (`yolov8n.pt`, auto-downloaded on first run) already detects **phone,
food and drink**. **Cigarette, seatbelt/no-seatbelt, and sunglasses are not COCO
classes** — you must train a model on your own labelled data for those.

## 1. Collect & label data
Gather in-cabin images covering each object in varied lighting, angles, and
drivers. Label them in **YOLO format** (one `.txt` per image, lines of
`class_id cx cy w h`, all normalised 0–1). Tools: Roboflow, CVAT, label-studio.

Class ids must match `training/data.yaml` (and `SAFETY_CLASS_MAP` in
`src/objects.py`):

```
0 cigarette   1 seatbelt   2 no_seatbelt   3 sunglasses   4 phone   5 food   6 drink
```

Layout:
```
training/dataset/
  images/train/*.jpg   images/val/*.jpg
  labels/train/*.txt   labels/val/*.txt
```

## 2. Train (GPU recommended)
```bash
pip install ultralytics
python training/train.py --data training/data.yaml --epochs 100 --device 0
```

## 3. Use the trained weights
Point the detector at your weights (e.g. in `src/config.py`):
```python
object_model_path = "runs/detect/train/weights/best.pt"
```
then run `python main.py`. Detections map onto safety categories automatically.

## Notes & honesty
- Model quality is **entirely** a function of dataset size/variety and training —
  budget for a few thousand labelled instances per class for usable accuracy.
- Seatbelt detection from a driver-facing camera is genuinely hard (occlusion,
  belt colour vs. clothing); expect to iterate.
- Validate on **your** camera and mounting position before trusting any alert.
