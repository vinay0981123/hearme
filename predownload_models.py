# predownload_models.py
# Best-effort script to pre-cache ASR & pyannote pipelines using the HF token.
import os
import traceback

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)
MODEL = os.getenv("TRANSCRIBE_MODEL", "small")
SEPARATION_PIPELINE = os.getenv("SEPARATION_PIPELINE", "pyannote/speech-separation-ami-1.0")
FALLBACK_DIAR_PIPELINE = os.getenv("SPEAKER_PIPELINE", "pyannote/speaker-diarization-3.1")

def try_whisperx():
    try:
        import whisperx
        print("[predownload] Loading whisperx model:", MODEL)
        runtime_device = "cuda" if (os.getenv("DEVICE", "cuda") != "cpu") else "cpu"
        # whisperx.load_model will download model weights to HF cache
        whisperx.load_model(MODEL, device=runtime_device)
        print("[predownload] whisperx model cached")
    except Exception as e:
        print("[predownload] whisperx predownload failed:", e)
        traceback.print_exc()

def try_faster_whisper():
    try:
        from faster_whisper import WhisperModel
        print("[predownload] Loading faster-whisper model:", MODEL)
        device = os.getenv("DEVICE", "cuda")
        compute = os.getenv("COMPUTE_TYPE", None) or ("float16" if device != "cpu" else "float32")
        WhisperModel(MODEL, device=device, compute_type=compute)
        print("[predownload] faster-whisper model cached")
    except Exception as e:
        print("[predownload] faster-whisper predownload failed:", e)
        traceback.print_exc()

def try_pyannote():
    try:
        from pyannote.audio import Pipeline
        print("[predownload] Downloading separation pipeline:", SEPARATION_PIPELINE)
        Pipeline.from_pretrained(SEPARATION_PIPELINE, use_auth_token=HF_TOKEN)
        print("[predownload] Downloading fallback diarization pipeline:", FALLBACK_DIAR_PIPELINE)
        Pipeline.from_pretrained(FALLBACK_DIAR_PIPELINE, use_auth_token=HF_TOKEN)
        print("[predownload] pyannote pipelines cached")
    except Exception as e:
        print("[predownload] pyannote predownload failed:", e)
        traceback.print_exc()

if __name__ == "__main__":
    print("[predownload] Starting predownload")
    try_whisperx()
    try_faster_whisper()
    try_pyannote()
    print("[predownload] Done")
