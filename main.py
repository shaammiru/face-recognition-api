from fastapi import FastAPI, UploadFile
from datetime import datetime
from deepface import DeepFace
import numpy as np
from PIL import Image
import io

app = FastAPI()

model_name = "SFace"
backend_detector = "opencv"


@app.get("/")
async def health_check():
    """
    Health check endpoint to verify if the API is running properly
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "FastAPI Health Check",
        "version": "1.0.0",
    }


@app.post("/recognize")
async def recognize_face(file: UploadFile):
    """
    Endpoint to handle face recognition requests
    """
    # convert img to numpy array
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)

    dfs = DeepFace.find(
        img_path=img_array,
        db_path="faces",
        model_name=model_name,
        detector_backend=backend_detector,
        align=True,
    )

    print(dfs)

    embedding = DeepFace.represent(
        img_path=img_array,
        model_name=model_name,
        detector_backend=backend_detector,
        align=True,
    )[0]["embedding"]

    print(f"Embedding: {len(embedding)}")

    return {
        "status": "success",
        "message": "Face recognition functionality is not yet implemented.",
        "file_name": file.filename,
    }


@app.post("/register")
async def register_face(file: UploadFile):
    """
    Endpoint to handle face registration requests
    """
    print(f"{file.size / 1024:.2f} KB")
    print(file.content_type)

    return {
        "status": "success",
        "message": "Face registration functionality is not yet implemented.",
        "file_name": file.filename,
    }
