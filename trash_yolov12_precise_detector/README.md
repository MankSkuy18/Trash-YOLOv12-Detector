# YOLOv12 Precise Trash Detector

This folder is for a real YOLOv12 object-detection model.

Important: precise detection needs bounding-box labels. The old datasets with
folders like `organic/`, `anorganic/`, and `B3/` are classification datasets;
they cannot train accurate object boxes by themselves.

## Classes

```text
0 organic
1 anorganic
2 B3
```

## Dataset Folder

Put YOLO detection images and labels here:

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  data.yaml
```

Each image must have a matching `.txt` label file.

Example:

```text
dataset/images/train/photo_001.jpg
dataset/labels/train/photo_001.txt
```

Label file format:

```text
class_id x_center y_center width height
```

All coordinates are normalized from `0` to `1`.

Example:

```text
1 0.512 0.430 0.322 0.280
```

Use a labeling tool such as Roboflow, CVAT, LabelImg, or makesense.ai, then
export as YOLO format.

## Train Detection Model

If you do not have manual bounding-box labels yet, create starter labels from
the existing 3-category classification dataset:

```powershell
python trash_detector.py bootstrap-from-classification --force
```

This writes one automatic box per image. It lets training run, but it is not as
precise as manually drawn boxes. Use this only as a starter model.

GPU:

```powershell
python trash_detector.py train --device 0 --epochs 60
```

CPU:

```powershell
python trash_detector.py train --device cpu --epochs 60
```

The trained model is saved at:

```text
runs/detect/trash-yolov12-detector/weights/best.pt
```

## Upload Photo and Predict

Open a file picker:

```powershell
python trash_detector.py predict-upload `
  --model runs/detect/trash-yolov12-detector/weights/best.pt `
  --device 0
```

Or pass an image path:

```powershell
python trash_detector.py predict `
  --model runs/detect/trash-yolov12-detector/weights/best.pt `
  --source C:\path\to\trash.jpg `
  --device 0
```

Results are saved to:

```text
predictions/
```
