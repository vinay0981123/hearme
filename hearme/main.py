# hearme/main.py
import os
import json
import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

from hearme.utils import download_public_url, standardize_audio
from hearme.transcribe import transcribe

load_dotenv()

# config via env
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "small")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", None)  # let transcribe decide best
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "2"))

# reduce uvicorn logging
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("fastapi").setLevel(logging.WARNING)

# FastAPI app — keep the variable name `app`
app = FastAPI(title="HearMe2 STT API", version="0.2")

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

class TranscribeRequest(BaseModel):
    s3_url: HttpUrl
    use_whisperx: bool = True   # optional


@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "service": "HearMe2 STT API", "version": "0.2"}


@app.post("/transcribe")
async def transcribe_endpoint(req: TranscribeRequest):
    try:
        # Use a per-request temp directory so nothing persists on disk after return
        with tempfile.TemporaryDirectory(prefix="stt_tmp_") as tmpdir:
            tmp_path = Path(tmpdir)

            # derive extension from URL path (ignore query)
            audio_url = str(req.s3_url)
            parsed = urlparse(audio_url)
            ext = Path(parsed.path).suffix or ".mp3"

            downloaded_path = tmp_path / ("input_audio" + ext)

            # 1) download
            try:
                download_public_url(audio_url, str(downloaded_path))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to download audio: {e}")

            # 2) standardize to 16k mono WAV in the same temp folder
            try:
                standardized_path = standardize_audio(
                    str(downloaded_path), 
                    str(tmp_path / "audio.wav"), 
                    sample_rate=16000
                )
            except subprocess.CalledProcessError as e:
                raise HTTPException(status_code=500, detail=f"Audio conversion failed: {e}")
            except FileNotFoundError:
                raise HTTPException(status_code=500, detail="ffmpeg not found on server. Please install ffmpeg.")

            # 3) run blocking processing in threadpool
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(
                    executor,
                    _process_job,
                    str(standardized_path),
                    req.use_whisperx
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

            return result

    except Exception as e:
        # Catch any unexpected errors and log them
        print("Unexpected error in /transcribe:", e)
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {e}")


def _process_job(audio_path: str, use_whisperx: bool):
    # Transcribe with diarization; no file writes
    transcript_segments = transcribe(
        audio_path,
        model_size=TRANSCRIBE_MODEL,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        use_whisperx=use_whisperx,
        hf_token=HUGGINGFACE_TOKEN
    )

    # Convert original speaker labels -> UserN in chronological order
    speaker_to_user = {}
    user_to_speaker = {}
    user_counter = 1
    mapped_user_segments = []
    for seg in sorted(transcript_segments, key=lambda s: s.get("start", 0.0)):
        sp = seg.get("speaker", "Unknown")
        if sp not in speaker_to_user:
            user_label = f"User{user_counter}"
            speaker_to_user[sp] = user_label
            user_to_speaker[user_label] = sp
            user_counter += 1
        mapped_user_segments.append({
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "speaker": speaker_to_user[sp],
            "text": seg.get("text", "").strip()
        })

    response = {
        "segments": mapped_user_segments,
        "mapping": user_to_speaker,
        "counts": {
            "segments_count": len(mapped_user_segments),
            "unique_speakers": len(user_to_speaker)
        }
    }
    return response
