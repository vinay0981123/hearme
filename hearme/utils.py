# app/utils.py
import warnings
import logging
from pathlib import Path
import requests
import subprocess

warnings.filterwarnings("ignore")
logging.getLogger("pyannote").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torchaudio").setLevel(logging.WARNING)
logging.getLogger("ffmpeg").setLevel(logging.WARNING)

def standardize_audio(in_path: str, out_path: str = None, sample_rate: int = 16000) -> str:
    in_p = Path(in_path)
    if out_path is None:
        out_path = str(in_p.with_suffix(".wav"))

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(in_p),
        "-ac", "1",
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        str(out_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr}")
    return str(out_path)


def download_public_url(url: str, dest_path: str, timeout: int = 120):
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return str(dest)
