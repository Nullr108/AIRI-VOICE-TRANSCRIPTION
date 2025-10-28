from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from faster_whisper import WhisperModel
from TTS.api import TTS
import soundfile as sf
import base64
import io
import uvicorn
from fastapi.middleware.cors import CORSMiddleware  # Для CORS

app = FastAPI()

# Добавляем CORS-middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Для теста; в проде укажите origin Airi, напр. ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем модели при старте
stt_model = WhisperModel("medium", device="cuda")  # Для STT
tts_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)  # XTTS-v2 для лучшего русского TTS

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
    # TTS с языком "ru" и сэмплом голоса
    wav = tts_model.tts(text=req.input, language="ru", speaker_wav="/app/sample.wav")
    buffer = io.BytesIO()
    sf.write(buffer, wav, 22050, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"audio": audio_b64}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
