from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
from PIL import Image
import numpy as np
import io
import os
import psycopg2
import base64
import time
import cv2

DB_CONN = psycopg2.connect(
    host="localhost",
    port=5432,
    database="faces_db",
    user="postgres",
    password="password",
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS = [
    "Dlib",
    "SFace",
    "OpenFace",
    "Facenet",
    "GhostFaceNet",
    "DeepID",
    "Facenet512",
    "ArcFace",
    "VGG-Face",
    "DeepFace",
]

BACKENDS = [
    "opencv",
    "ssd",
    "dlib",
    "mtcnn",
    "fastmtcnn",
    "retinaface",
    "mediapipe",
    "yolov8",
    "yolov11s",
    "yolov11n",
    "yolov11m",
    "yunet",
    "centerface",
]

EMBED_DIM = 512
L2_THRESHOLD = 1.0


def prepare_image(pil_img, max_dim=640):
    pil_img = pil_img.convert("RGB")

    if max(pil_img.size) > max_dim:
        pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)

    img_array = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.bilateralFilter(img_bgr, 5, 50, 50)

    return img_bgr


def to_vector_str(vec: np.ndarray):
    return "[" + ",".join(str(float(x)) for x in vec.tolist()) + "]"


@app.get("/faces")
async def list_faces():
    with DB_CONN.cursor() as cur:
        cur.execute("SELECT id, name, embedding, image_base FROM faces;")
        rows = cur.fetchall()

    faces = []
    for row in rows:
        face_id, name, embedding, image_base = row
        try:
            emb_len = len(embedding)
        except Exception:
            emb_len = None
        faces.append(
            {
                "id": face_id,
                "name": name,
                "embedding_length": emb_len,
                "image_base": image_base,
            }
        )

    return {"faces": faces}


@app.delete("/faces/{face_id}")
async def delete_face(face_id: int):
    with DB_CONN.cursor() as cur:
        cur.execute("SELECT * FROM faces WHERE id = %s;", (face_id,))
        if cur.fetchone() is None:
            raise HTTPException(status_code=404, detail="Face not found.")

        cur.execute("DELETE FROM faces WHERE id = %s;", (face_id,))
        DB_CONN.commit()

    return {"status": "success", "message": f"Face with id {face_id} deleted."}


@app.post("/recognize", status_code=200)
async def recognize_face(file: UploadFile):
    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image")

    img_array = prepare_image(image, max_dim=640)

    if not os.path.exists("logs"):
        os.makedirs("logs")
    generated_filename = f"log_{int(time.time())}.jpg"
    image.save(f"logs/{generated_filename}")

    start_time = time.time()

    try:
        rep = DeepFace.represent(
            img_path=img_array,
            model_name=MODELS[7],
            detector_backend=BACKENDS[3],
            normalization="ArcFace",
            enforce_detection=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"face represent error: {str(e)}")

    if not rep:
        raise HTTPException(status_code=404, detail="no face detected")

    embedding = np.array(rep[0]["embedding"], dtype=np.float32)

    norm = np.linalg.norm(embedding)
    if norm != 0:
        embedding = embedding / norm

    vector_str = to_vector_str(embedding)

    with DB_CONN.cursor() as cur:
        cur.execute(
            """
            SELECT name, embedding <-> %s AS distance
            FROM faces
            ORDER BY embedding <-> %s
            LIMIT 1;
            """,
            (vector_str, vector_str),
        )
        result = cur.fetchone()

    if not result:
        raise HTTPException(status_code=404, detail="no faces in database")

    name, distance = result
    time_taken = time.time() - start_time
    print(f"Recognized face: {name}, Distance: {distance:.4f} (took {time_taken:.3f}s)")

    if distance <= L2_THRESHOLD:
        return {"status": "success", "match": name, "distance": round(distance, 4)}
    else:
        raise HTTPException(
            status_code=404,
            detail=f"No match (distance {distance:.4f} with face {name} exceeds threshold {L2_THRESHOLD:.4f})",
        )


@app.post("/register", status_code=201)
async def register_face(file: UploadFile = File(...), name: str = Form(...)):
    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image")

    img_array = prepare_image(image, max_dim=640)

    try:
        rep = DeepFace.represent(
            img_path=img_array,
            model_name=MODELS[7],
            detector_backend=BACKENDS[3],
            normalization="ArcFace",
            enforce_detection=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"face represent error: {str(e)}")

    if not rep:
        raise HTTPException(status_code=400, detail="no face found in image")

    embedding = np.array(rep[0]["embedding"], dtype=np.float32)

    norm = np.linalg.norm(embedding)
    if norm != 0:
        embedding = embedding / norm

    vector_str = to_vector_str(embedding)
    image_base64 = base64.b64encode(contents).decode("utf-8")

    with DB_CONN.cursor() as cur:
        cur.execute(
            "INSERT INTO faces (name, embedding, image_base) VALUES (%s, %s, %s)",
            (name, vector_str, image_base64),
        )
        DB_CONN.commit()

    return {
        "status": "success",
        "message": f"Face '{name}' registered.",
        "embedding_size": len(embedding),
    }
