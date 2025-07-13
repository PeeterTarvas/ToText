import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, JSONResponse  # Changed to include Response
import numpy as np
import cv2
import os
from ultralytics import YOLO
from collections import defaultdict
from PIL import Image

app = FastAPI()

def load_model(weights_path):
    print(f"[INFO] Loading model from {weights_path}")
    return YOLO(weights_path)

def run_inference(model, image_path):
    print(f"[INFO] Running inference on {image_path}")
    results = model(image_path)
    return results[0]

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
        line = sorted(lines_dict[line_idx], key=lambda x: x[0])
        text_line = ''.join(char for _, char in line)
        lines_text.append(text_line)
    return lines_text

model_weights = "../training/results/run1/last.pt"
model = load_model(model_weights)

@app.post("/detect/")
async def detect_image(file: UploadFile = File(...)):
    try:
        image_path = f"uploads/{file.filename}"
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = run_inference(model, image_path)
        boxes = result.boxes
        class_names = model.names

        lines = extract_sorted_characters(boxes, class_names)
        text_lines = reconstruct_text(lines)

        os.remove(image_path)

        return JSONResponse(content={"text": text_lines})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/detect_bboxes/")
async def detect_bboxes(file: UploadFile = File(...)):
    try:
        image_path = f"uploads/{file.filename}"
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = run_inference(model, image_path)
        boxes = result.boxes
        img_width, img_height = result.orig_shape[1], result.orig_shape[0]

        yolo_lines = []
        for box in boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

            cx = (x1 + x2) / (2 * img_width)
            cy = (y1 + y2) / (2 * img_height)
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height

            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")

        os.remove(image_path)

        return Response(
            content="\n".join(yolo_lines),
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=detections.txt"}
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    uvicorn.run(app, host="0.0.0.0", port=8000)