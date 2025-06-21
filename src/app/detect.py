# yolo_emnist_ocr.py

from ultralytics import YOLO
from collections import defaultdict
import numpy as np
from PIL import Image
import sys

def load_model(weights_path):
    print(f"[INFO] Loading model from {weights_path}")
    return YOLO(weights_path)

def run_inference(model, image_path):
    print(f"[INFO] Running inference on {image_path}")
    results = model(image_path)

    # Save annotated image
    results[0].save(filename="annotated_output.jpg")
    print("[INFO] Annotated image saved as 'annotated_output.jpg'")

    return results[0].boxes, model.names

def extract_sorted_characters(boxes, class_names, line_height=40):
    print("[INFO] Extracting and sorting characters")
    detections = []

    for box in boxes:
        cls_id = int(box.cls[0])
        char = class_names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        detections.append((char, cx, cy))

    lines = defaultdict(list)
    for char, cx, cy in detections:
        line_idx = round(cy / line_height)
        lines[line_idx].append((cx, char))

    return lines

def reconstruct_text(lines_dict):
    print("[INFO] Reconstructing text layout")
    lines_text = []
    for line_idx in sorted(lines_dict.keys()):
        line = sorted(lines_dict[line_idx], key=lambda x: x[0])  # sort by x
        text_line = ''.join(char for _, char in line)
        lines_text.append(text_line)
    return lines_text

def save_to_file(text_lines, output_path="output.txt"):
    print(f"[INFO] Saving to {output_path}")
    with open(output_path, "w") as f:
        for line in text_lines:
            f.write(line + "\n")

def main(image_path, weights_path):
    model = load_model(weights_path)
    boxes, class_names = run_inference(model, image_path)
    lines = extract_sorted_characters(boxes, class_names, line_height=40)
    text_lines = reconstruct_text(lines)
    save_to_file(text_lines)

    print("[✅] Done! Output written to 'output.txt'")

# If running from CLI:
if __name__ == "__main__":
    #if len(sys.argv) != 3:
    #    print("Usage: python yolo_emnist_ocr.py <image_path> <weights_path>")
    #else:
        #image_path = sys.argv[1]
        #weights_path = sys.argv[2]
    image_path = "src/training/results/run1/rubric.png"
    weights_path = "training/results/run1/iteration1_best.pt"
    main(image_path, weights_path)
