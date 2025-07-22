from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from datetime import datetime
from deepface import DeepFace
from PIL import Image
import numpy as np
import io
import psycopg2

DB_CONN = psycopg2.connect(
    host="localhost",
    port=5432,
    database="faces_db",
    user="postgres",
    password="password",
)

app = FastAPI()

MODEL_NAME = "SFace"
DETECTOR = "opencv"


@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "FastAPI Health Check",
        "version": "1.0.0",
    }


@app.get("/faces")
async def list_faces():
    with DB_CONN.cursor() as cur:
        cur.execute("SELECT id, name, embedding FROM faces;")
        rows = cur.fetchall()

    faces = []
    for row in rows:
        face_id, name, embedding = row
        faces.append(
            {
                "id": face_id,
                "name": name,
                "embedding_length": len(embedding) if embedding else 0,
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

    embedding = DeepFace.represent(
        img_path=img_array,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=True,
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
        if distance < -0.6:
            return {"status": "success", "match": name, "distance": round(distance, 3)}

        raise HTTPException(
            status_code=404, detail="No matching face found with sufficient confidence."
        )

    raise HTTPException(status_code=404, detail="No matching face found.")


@app.post("/register", status_code=201)
async def register_face(file: UploadFile = File(...), name: str = Form(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_array = np.array(image)

    result = DeepFace.represent(
        img_path=img_array,
        model_name=MODEL_NAME,
        detector_backend=DETECTOR,
        enforce_detection=True,
    )[0]["embedding"]

    result = np.array(result)
    norm = np.linalg.norm(result)

    if norm != 0:
        result = result / norm

    with DB_CONN.cursor() as cur:
        cur.execute(
            "INSERT INTO faces (name, embedding) VALUES (%s, %s)",
            (name, result.tolist()),
        )
        DB_CONN.commit()

    return {
        "status": "success",
        "message": "Face registered successfully.",
    }
