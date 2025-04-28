from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import pipeline
import io
import numpy as np
from typing import List

od_pipe = pipeline("object-detection", model="hustvl/yolos-tiny")

app = FastAPI()


def detect_objects(image: np.ndarray) -> List[str]:
    pil_image = Image.fromarray(image)
    pipeline_output = od_pipe(pil_image)

    results = []
    for result in pipeline_output:
        label = result["label"]  # название объекта
        score = result["score"]  # вероятность
        results.append(f"{label}: {score:.2f}")
    return results


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    image_np = np.array(image)

    results = detect_objects(image_np)

    return {"results": results}
