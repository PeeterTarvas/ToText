import shutil

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import pymupdf
import numpy as np
import cv2
import os

app = FastAPI()

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save the uploaded PDF
        pdf_path = f"uploads/{file.filename}"
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the PDF and extract images
        images = convert_pdf_to_images(pdf_path)

        # Process each image and recognize letters (this part should use your existing recognition logic)
        results = []
        for img in images:
            recognized_text = recognize_letters_from_image(img)
            results.append(recognized_text)

        return JSONResponse(content={"results": results})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def convert_pdf_to_images(pdf_path):
    # Use PyMuPDF to extract images from PDF
    doc = pymupdf.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img = np.array(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        images.append(img)
    return images

def recognize_letters_from_image(img):
    # Placeholder function for recognizing letters from an image
    # Replace this with your actual letter recognition logic
    return "recognized_text"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)