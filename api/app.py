from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multimodal_emotion import MultiModalEmotionDetector

app = FastAPI(title="Multi-Modal Emotion Detection API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = MultiModalEmotionDetector()

class TextInput(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Multi-Modal Emotion Detection API"}

@app.post("/predict/text")
async def predict_text_emotion(input_data: TextInput):
    try:
        result = detector.predict_text(input_data.text)
        return {"emotions": result, "predicted_emotion": max(result, key=result.get)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/audio")
async def predict_audio_emotion(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        result = detector.predict_audio(tmp_file_path)
        os.unlink(tmp_file_path)

        return {"emotions": result, "predicted_emotion": max(result, key=result.get)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/image")
async def predict_image_emotion(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        result = detector.predict_image(tmp_file_path)
        os.unlink(tmp_file_path)

        return {"emotions": result, "predicted_emotion": max(result, key=result.get)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/multimodal")
async def predict_multimodal_emotion(
    text: Optional[str] = None,
    audio_file: Optional[UploadFile] = File(None),
    image_file: Optional[UploadFile] = File(None)
):
    try:
        audio_path = None
        image_path = None

        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                content = await audio_file.read()
                tmp_file.write(content)
                audio_path = tmp_file.name

        if image_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                content = await image_file.read()
                tmp_file.write(content)
                image_path = tmp_file.name

        result = detector.predict_multimodal(
            text=text,
            audio=audio_path,
            image=image_path
        )

        if audio_path:
            os.unlink(audio_path)
        if image_path:
            os.unlink(image_path)

        return {"emotions": result, "predicted_emotion": max(result, key=result.get)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
