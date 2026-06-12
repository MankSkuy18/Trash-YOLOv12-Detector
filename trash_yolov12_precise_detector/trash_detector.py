"""Train and run a precise YOLOv12 trash object detector.

This script expects a YOLO detection dataset with bounding-box labels.
Classes:
  0 organic
  1 anorganic
  2 B3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path


DEFAULT_DATA = Path("dataset/data.yaml")
DEFAULT_CLASSIFICATION_DATA = Path("../trash_sorter_workspace/trash-3-category-yolo-cls")
DEFAULT_MODEL = "yolov12n.pt"
MODEL_DIR = Path("models")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_TO_ID = {"organic": 0, "anorganic": 1, "B3": 2}

YOLOV12_DETECTION_URLS = {
    "yolov12n.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12n.pt",
    "yolov12s.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12s.pt",
    "yolov12m.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12m.pt",
    "yolov12l.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12l.pt",
    "yolov12x.pt": "https://github.com/sunsmarterjie/yolov12/releases/download/turbo/yolov12x.pt",
}


def load_yolo():
    try:
        import numpy as np

        if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
            np.trapz = np.trapezoid
    except ImportError:
        pass

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Missing YOLOv12 dependencies. From the main project folder, run:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc
    return YOLO


def resolve_model(model: str, allow_download: bool) -> str:
    if model.startswith(("http://", "https://")):
        return model

    model_path = Path(model).expanduser()
    if model_path.exists():
        return str(model_path)

    model_name = model_path.name
    url = YOLOV12_DETECTION_URLS.get(model_name)
    if allow_download and url:
        target = MODEL_DIR / model_name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            print(f"Downloading {model_name} ...")
            urllib.request.urlretrieve(url, target)
        return str(target)

    raise FileNotFoundError(
        f"Model file not found: {model}. Use a trained best.pt or one of "
        f"{', '.join(sorted(YOLOV12_DETECTION_URLS))}."
    )


def validate_dataset(data_yaml: Path) -> None:
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    root = data_yaml.parent
    required = [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing YOLO detection folders:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )

    train_labels = list((root / "labels" / "train").glob("*.txt"))
    val_labels = list((root / "labels" / "val").glob("*.txt"))
    if not train_labels or not val_labels:
        raise FileNotFoundError(
            "No bounding-box label files found. Add YOLO .txt labels before "
            "training. Classification folders alone cannot train detection."
        )


def write_runtime_data_yaml(data_yaml: Path) -> Path:
    dataset_root = data_yaml.parent.resolve()
    runtime_yaml = dataset_root / "data.runtime.yaml"
    runtime_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                "  0: organic",
                "  1: anorganic",
                "  2: B3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return runtime_yaml


def iter_images(folder: Path):
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def clean_detection_dataset(dataset_root: Path) -> None:
    for split in ("train", "val", "test"):
        for section in ("images", "labels"):
            folder = dataset_root / section / split
            if folder.exists():
                shutil.rmtree(folder)
            folder.mkdir(parents=True, exist_ok=True)


def box_for_mode(mode: str) -> str:
    if mode == "full":
        return "0.500000 0.500000 1.000000 1.000000"
    if mode == "center":
        return "0.500000 0.500000 0.700000 0.700000"
    raise ValueError(f"Unknown box mode: {mode}")


def bootstrap_from_classification(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser().resolve()
    dataset_root = Path(args.dataset).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Classification dataset not found: {source}")

    if args.force:
        clean_detection_dataset(dataset_root)
    else:
        for split in ("train", "val", "test"):
            (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    label_box = box_for_mode(args.box_mode)
    summary = {}
    for split in ("train", "val", "test"):
        split_root = source / split
        if not split_root.exists():
            continue

        summary[split] = {}
        for class_name, class_id in CLASS_TO_ID.items():
            class_root = split_root / class_name
            if not class_root.exists():
                summary[split][class_name] = 0
                continue

            count = 0
            for image_path in iter_images(class_root):
                safe_name = f"{split}_{class_name}_{image_path.stem}{image_path.suffix.lower()}"
                target_image = dataset_root / "images" / split / safe_name
                target_label = dataset_root / "labels" / split / f"{Path(safe_name).stem}.txt"
                shutil.copy2(image_path, target_image)
                target_label.write_text(f"{class_id} {label_box}\n", encoding="utf-8")
                count += 1
            summary[split][class_name] = count

    summary_path = dataset_root / "bootstrap_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Created YOLO detection labels from the classification dataset.")
    print(f"Dataset path: {dataset_root}")
    print(f"Box mode: {args.box_mode}")
    for split, classes in summary.items():
        total = sum(classes.values())
        print(f"  - {split}: {total} images")
        for class_name, count in classes.items():
            print(f"      {class_name}: {count}")
    print(
        "Note: generated boxes are automatic. For truly precise detection, "
        "replace them with manually annotated boxes later."
    )


def train(args: argparse.Namespace) -> None:
    YOLO = load_yolo()
    data_yaml = Path(args.data).expanduser()
    validate_dataset(data_yaml)
    runtime_data_yaml = write_runtime_data_yaml(data_yaml)

    model = YOLO(resolve_model(args.model, allow_download=True))
    results = model.train(
        data=str(runtime_data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )
    print(results)


def choose_image_file() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_name = filedialog.askopenfilename(
        title="Choose a trash image",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return Path(file_name) if file_name else None


def predict_image(args: argparse.Namespace, upload: bool = False) -> None:
    YOLO = load_yolo()
    image_path = choose_image_file() if upload else Path(args.source).expanduser()
    if image_path is None:
        print("No image selected.")
        return
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output = Path(args.output).expanduser()
    output.mkdir(parents=True, exist_ok=True)

    model = YOLO(resolve_model(args.model, allow_download=False))
    results = model.predict(
        source=str(image_path),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        project=str(output),
        name=image_path.stem,
        exist_ok=True,
        verbose=False,
    )

    rows = []
    result = results[0]
    if result.boxes is not None:
        for index, box in enumerate(result.boxes, start=1):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = [round(float(v), 2) for v in box.xyxy[0]]
            rows.append(
                {
                    "object": index,
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": round(confidence, 4),
                    "box_xyxy": xyxy,
                }
            )

    report_path = output / image_path.stem / "prediction_report.json"
    report_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Predicted {len(rows)} object(s).")
    for row in rows:
        print(f"  object {row['object']}: {row['class_name']} {row['confidence']:.0%}")
    print(f"Saved annotated image folder: {output / image_path.stem}")
    print(f"Saved report: {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLOv12 precise trash object detector."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train detection model")
    train_parser.add_argument("--data", default=str(DEFAULT_DATA))
    train_parser.add_argument("--model", default=DEFAULT_MODEL)
    train_parser.add_argument("--epochs", type=int, default=60)
    train_parser.add_argument("--imgsz", type=int, default=640)
    train_parser.add_argument("--batch", type=int, default=16)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--workers", type=int, default=4)
    train_parser.add_argument("--patience", type=int, default=15)
    train_parser.add_argument("--project", default="runs/detect")
    train_parser.add_argument("--name", default="trash-yolov12-detector")

    bootstrap_parser = subparsers.add_parser(
        "bootstrap-from-classification",
        help="create starter YOLO detection labels from organic/anorganic/B3 folders",
    )
    bootstrap_parser.add_argument("--source", default=str(DEFAULT_CLASSIFICATION_DATA))
    bootstrap_parser.add_argument("--dataset", default="dataset")
    bootstrap_parser.add_argument(
        "--box-mode",
        choices=["full", "center"],
        default="full",
        help="automatic box style to write into YOLO label files",
    )
    bootstrap_parser.add_argument(
        "--force",
        action="store_true",
        help="clear existing dataset/images and dataset/labels first",
    )

    predict_parser = subparsers.add_parser("predict", help="predict an image path")
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--source", required=True)
    predict_parser.add_argument("--output", default="predictions")
    predict_parser.add_argument("--imgsz", type=int, default=640)
    predict_parser.add_argument("--conf", type=float, default=0.25)
    predict_parser.add_argument("--device", default="cpu")

    upload_parser = subparsers.add_parser(
        "predict-upload", help="choose an image and predict"
    )
    upload_parser.add_argument("--model", required=True)
    upload_parser.add_argument("--output", default="predictions")
    upload_parser.add_argument("--imgsz", type=int, default=640)
    upload_parser.add_argument("--conf", type=float, default=0.25)
    upload_parser.add_argument("--device", default="cpu")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "train":
            train(args)
        elif args.command == "bootstrap-from-classification":
            bootstrap_from_classification(args)
        elif args.command == "predict":
            predict_image(args)
        elif args.command == "predict-upload":
            predict_image(args, upload=True)
        else:
            parser.error(f"Unknown command: {args.command}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
