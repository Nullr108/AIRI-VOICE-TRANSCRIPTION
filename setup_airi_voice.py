import os
from pathlib import Path

# Корневая папка проекта
root = Path("airi-voice")
app_dir = root / "app"

# Структура папок
app_dir.mkdir(parents=True, exist_ok=True)

# docker-compose.yml
docker_compose = """version: "3.9"

services:
  airi-voice:
    build: ./app
    container_name: airi-voice
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
"""

# Dockerfile
dockerfile = """FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Базовые зависимости
RUN apt-get update && apt-get install -y \\
    python3 python3-pip ffmpeg git libsndfile1 && \\
    rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN pip3 install --upgrade pip

# Устанавливаем зависимости для API
RUN pip3 install fastapi uvicorn[standard] faster-whisper TTS soundfile

WORKDIR /app
COPY app.py /app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# app.py
app_py = """from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from faster_whisper import WhisperModel
from TTS.api import TTS
import soundfile as sf
import base64
import io
import uvicorn

app = FastAPI()

# Загружаем модели при старте
stt_model = WhisperModel("small", device="cuda")  # можно "medium" для качества
tts_model = TTS("tts_models/ru/v3_1_0", gpu=True)  # Русский голос Coqui TTS

# ----- STT -----
@app.post("/v1/audio/transcriptions")
async def transcribe(file: UploadFile):
    audio_path = "/tmp/temp.wav"
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    segments, _ = stt_model.transcribe(audio_path)
    text = " ".join([seg.text for seg in segments])
    return {"text": text}

# ----- TTS -----
class TTSRequest(BaseModel):
    model: str = "gpt-tts"
    input: str

@app.post("/v1/audio/speech")
def speech(req: TTSRequest):
    wav = tts_model.tts(req.input)
    buffer = io.BytesIO()
    sf.write(buffer, wav, 22050, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"data": [{"b64_json": audio_b64}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

# Создание файлов
(root / "docker-compose.yml").write_text(docker_compose, encoding="utf-8")
(app_dir / "Dockerfile").write_text(dockerfile, encoding="utf-8")
(app_dir / "app.py").write_text(app_py, encoding="utf-8")

print("✅ Проект создан в папке airi-voice/")
print("Теперь запусти:")
print("    cd airi-voice")
print("    docker-compose up --build")
