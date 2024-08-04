import os
import numpy as np
import gzip
import shutil
import cv2


def normalize_bbox(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height


os.makedirs('yolo_dataset/images/train', exist_ok=True)
os.makedirs('yolo_dataset/labels/train', exist_ok=True)

gz_file = './data/emnist-letters-train-images-idx3-ubyte.gz'
gz_file_label = './data/emnist-letters-train-labels-idx1-ubyte.gz'
with gzip.open(gz_file, 'rb') as f:
    x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
with gzip.open(gz_file_label, 'rb') as f:
    y_train = np.frombuffer(f.read(), np.uint8, offset=8)

img_width, img_height = 28, 28

for i, (img, label) in enumerate(zip(x_train, y_train)):
    img_file_path = f'yolo_dataset/images/train/{i}.png'
    cv2.imwrite(img_file_path, img)

    bbox = [0, 0, img_width, img_height]
    x_center, y_center, width, height = normalize_bbox(bbox, img_width, img_height)

    label_file_path = f'yolo_dataset/labels/train/{i}.txt'
    with open(label_file_path, 'w') as f:
        f.write(f'{label - 1} {x_center} {y_center} {width} {height}\n')

print("Dataset preparation complete!")