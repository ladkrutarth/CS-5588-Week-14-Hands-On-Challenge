"""
Phase 2 API layer (FastAPI) for STT + TTS.

Run:
uvicorn api_server:app --host 0.0.0.0 --port 9000 --reload
"""

from __future__ import annotations

import io
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from src.models.whisper_model import WhisperASR
from src.utils.audio_utils import load_audio
from src.utils.tts_engine import LocalTTSEngine


app = FastAPI(title="AI Speech Intelligence API", version="2.0")

_whisper: Optional[WhisperASR] = None
_tts: Optional[LocalTTSEngine] = None


def get_whisper() -> WhisperASR:
    global _whisper
    if _whisper is None:
        _whisper = WhisperASR()
        _whisper.load_model()
    return _whisper


def get_tts() -> LocalTTSEngine:
    global _tts
    if _tts is None:
        _tts = LocalTTSEngine()
    return _tts


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/stt")
async def speech_to_text(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe"),
):
    try:
        audio_bytes = await audio.read()
        waveform, sr = load_audio(audio_bytes)
        whisper = get_whisper()
        result = whisper.transcribe(
            waveform,
            sampling_rate=sr,
            language=language,
            task=task,
        )
        return JSONResponse(
            {
                "text": result["text"],
                "language": result["language"],
                "latency_seconds": result["latency_seconds"],
                "device": result["device"],
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT failed: {e}")


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    rate: int = Form(170),
    voice_id: Optional[str] = Form(None),
):
    try:
        tts = get_tts()
        wav_bytes = tts.synthesize_to_wav_bytes(text, rate=rate, voice_id=voice_id)
        return StreamingResponse(
            io.BytesIO(wav_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tts_output.wav"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")
