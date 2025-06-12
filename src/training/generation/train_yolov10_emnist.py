from ultralytics import YOLO
import yaml
from pathlib import Path

DATA_ROOT = Path("synthetic_emnist_yolo")
DATA_YAML_PATH = Path("emnist.yaml")
NUM_CLASSES = 47
IMG_SIZE = 640
EPOCHS = 1
MODEL_ARCH = "yolov10s.pt"

CHAR_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!?.,:;'\"()[]{}<>@#$%^&*+-=/\\|_"
CHAR_MAP = CHAR_MAP[:NUM_CLASSES]
class_names = list(CHAR_MAP)

data_yaml = {
    'path': str(DATA_ROOT),
    'train': 'images/train',
    'val': 'images/val',
    'nc': NUM_CLASSES,
    'names': class_names
}

with open(DATA_YAML_PATH, 'w') as f:
    yaml.dump(data_yaml, f)

print(f"[INFO] Dataset config written to {DATA_YAML_PATH.resolve()}")

model = YOLO(MODEL_ARCH)

model.train(
    data=str(DATA_YAML_PATH),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    project="runs/train_emnist_yolo",
    name="yolov10s_emnist",
    batch=16,
    workers=4,
    device=0
)

print("[INFO] Training completed.")
