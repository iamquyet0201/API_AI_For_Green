from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Any
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import logging
from io import BytesIO
import base64
from rembg import remove

# Logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Class mapping: 6 lớp
class_mapping = {
    0: 'plastic_bottle',
    1: 'plastic_bottle_cap',
    2: 'paper_cup',
    3: 'tongue_depressor',
    4: 'cardboard',
    5: 'straw'
}

# FastAPI app
app = FastAPI()

# Enable CORS (frontend can call)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
try:
    model = YOLO('best.pt')
except Exception as e:
    logging.error(f"Could not load model: {e}")
    raise RuntimeError("Model load failed")

# PIL ↔ OpenCV conversions
def pil_to_cv2(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Convert image to base64
def encode_image_to_base64(image: Image.Image) -> str:
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Convert numpy to Python native types
def convert_numpy_types(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    return obj

# Remove background and paste on white
def remove_background_and_paste_on_white(image: Image.Image) -> Image.Image:
    try:
        no_bg = remove(image)
        no_bg = Image.open(BytesIO(no_bg)).convert("RGBA")
    except Exception as e:
        logging.error(f"Background removal failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove background")

    white_bg = Image.new("RGBA", no_bg.size, (255, 255, 255, 255))
    combined = Image.alpha_composite(white_bg, no_bg)
    return combined.convert("RGB")

# Root route
@app.get("/")
def root():
    return {"message": "API is working!"}

# Detection route
@app.post("/det")
async def detection(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(BytesIO(image_data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Cannot read image")

    # ❄️ Remove background and paste on white
    image = remove_background_and_paste_on_white(image)

    # 🧠 YOLOv8 detection
    results = model(source=image, conf=0.3, iou=0.5)
    img_cv2 = pil_to_cv2(image)
    class_counts = {}
    det = [0] * len(class_mapping)

    for result in results:
        boxes = result.boxes.cpu().numpy()
        xyxy = boxes.xyxy
        cls_ids = boxes.cls.astype(int)

        for box, cls_id in zip(xyxy, cls_ids):
            cv2.rectangle(img_cv2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            class_name = class_mapping.get(cls_id, f"Unknown({cls_id})")
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            if cls_id < len(det):
                det[cls_id] += 1

    # ✨ Encode images
    img_result = cv2_to_pil(img_cv2)
    base64_original = encode_image_to_base64(image)
    base64_result = encode_image_to_base64(img_result)

    # 📦 Return result
    return {
        "data": {
            "base64_r": base64_result,
            "class_mapping": class_mapping,
            "result": {
                "dict": class_counts,
                "det": det,
            },
        },
        "msg": "success",
        "code": 200
    }
