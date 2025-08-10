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
DETECTOR = "opencv"

# print all models embedding size
for model in MODELS:
    embedding_size = DeepFace.build_model(model).output_shape[1]
    print(f"Model: {model}, Embedding Size: {embedding_size}")


@app.get("/faces")
async def list_faces():
    with DB_CONN.cursor() as cur:
        cur.execute("SELECT id, name, embedding, image_base FROM faces;")
        rows = cur.fetchall()

    faces = []
    for row in rows:
        face_id, name, embedding, image_base = row
        faces.append(
            {
                "id": face_id,
                "name": name,
                "embedding_length": len(embedding) if embedding else 0,
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
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)

    if not os.path.exists("logs"):
        os.makedirs("logs")

    img_path = f"logs/{file.filename}"
    image.save(img_path)

    start_time = time.time()

    embedding = DeepFace.represent(
        img_path=img_array,
        model_name=MODELS[0],
        detector_backend=DETECTOR,
        enforce_detection=False,
    )[0]["embedding"]

    embedding = np.array(embedding)
    norm = np.linalg.norm(embedding)
    if norm != 0:
        embedding = embedding / norm

    with DB_CONN.cursor() as cur:
        vector_str = "[" + ",".join(str(x) for x in embedding) + "]"

        cur.execute(
            """
            SELECT name, embedding <#> %s AS distance
            FROM faces
            ORDER BY embedding <#> %s
            LIMIT 1;
            """,
            (vector_str, vector_str),
        )

        result = cur.fetchone()

    if result:
        name, distance = result

        time_taken = time.time() - start_time
        print(f"Recognition took {time_taken:.2f} seconds.")

        if distance < -0.4:
            return {"status": "success", "match": name, "distance": round(distance, 3)}

        raise HTTPException(
            status_code=404,
            detail="No matching face found with sufficient confidence, distance: {:.3f}".format(
                distance
            ),
        )

    raise HTTPException(status_code=404, detail="No matching face found.")


@app.post("/register", status_code=201)
async def register_face(file: UploadFile = File(...), name: str = Form(...)):
    contents = await file.read()
    image_base64 = base64.b64encode(contents).decode("utf-8")
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)

    result = DeepFace.represent(
        img_path=img_array,
        model_name=MODELS[0],
        detector_backend=DETECTOR,
        enforce_detection=False,
    )[0]["embedding"]

    print(f"Embedding size: {len(result)}")

    result = np.array(result)
    norm = np.linalg.norm(result)

    if norm != 0:
        result = result / norm

    with DB_CONN.cursor() as cur:
        cur.execute(
            "INSERT INTO faces (name, embedding, image_base) VALUES (%s, %s, %s)",
            (name, result.tolist(), image_base64),
        )
        DB_CONN.commit()

    return {
        "status": "success",
        "message": "Face registered successfully.",
    }
