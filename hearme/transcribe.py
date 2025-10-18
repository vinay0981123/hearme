# app/transcribe.py
from typing import List, Dict, Optional
import os

def _group_words_to_segments(words, gap_threshold=0.9):
    """
    words: list of {"start": float, "end": float, "word": str, "speaker": str}
    groups contiguous words with same speaker into segments (merge by gap threshold).
    """
    if not words:
        return []

    groups = []
    cur = {"start": None, "end": None, "speaker": None, "words": []}
    for w in words:
        sp = w.get("speaker", "Unknown")
        s = float(w.get("start", 0.0))
        e = float(w.get("end", s))
        token = w.get("word", "").strip()

        if cur["start"] is None:
            cur = {"start": s, "end": e, "speaker": sp, "words": [token] if token else []}
        else:
            gap = s - cur["end"]
            if sp == cur["speaker"] and gap <= gap_threshold:
                cur["end"] = e
                if token:
                    cur["words"].append(token)
            else:
                text = " ".join(cur["words"]).strip()
                groups.append({"start": cur["start"], "end": cur["end"], "speaker": cur["speaker"], "text": text})
                cur = {"start": s, "end": e, "speaker": sp, "words": [token] if token else []}

    if cur["start"] is not None:
        text = " ".join(cur["words"]).strip()
        groups.append({"start": cur["start"], "end": cur["end"], "speaker": cur["speaker"], "text": text})
    return groups


def _map_words_to_speakers_by_diarization(word_segments, diarization_segments):
    """
    Map each word (with start,end) to a diarization speaker by midpoint overlap.
    word_segments: list of {"start","end","word"}
    diarization_segments: list of {"start","end","speaker"} (sorted)
    Returns words list with 'speaker' filled.
    """
    mapped = []
    diarization = sorted(diarization_segments, key=lambda d: d["start"])
    for w in word_segments:
        ws = float(w.get("start", 0.0))
        we = float(w.get("end", ws))
        mid = (ws + we) / 2.0
        chosen = None
        for d in diarization:
            if d["start"] - 0.01 <= mid <= d["end"] + 0.01:
                chosen = d["speaker"]
                break
        if chosen is None:
            best = None
            best_ratio = 0.0
            dur = we - ws if we > ws else 1.0
            for d in diarization:
                overlap = max(0.0, min(we, d["end"]) - max(ws, d["start"]))
                ratio = overlap / dur
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = d["speaker"]
            chosen = best if best is not None else "Unknown"
        mapped.append({"start": ws, "end": we, "word": w.get("word", ""), "speaker": chosen})
    return mapped


def transcribe(audio_path: str,
               model_size: str = "small",
               device: str = "cpu",
               compute_type: Optional[str] = None,
               use_whisperx: bool = True,
               hf_token: Optional[str] = None) -> List[Dict]:
    """
    Pipeline:
      1) WhisperX (transcribe + align words).
      2) Pyannote separation+diarization (speech-separation-ami-1.0).
      3) Map words -> diarization speakers; group to readable segments.
      Fallbacks: pyannote speaker-diarization-3.1 with fixed 2 speakers, then ASR-only.
    """
    if hf_token is None:
        hf_token = os.getenv("HUGGINGFACE_TOKEN", None)

    SEPARATION_PIPELINE = os.getenv("SEPARATION_PIPELINE", "pyannote/speech-separation-ami-1.0")
    FALLBACK_DIAR_PIPELINE = os.getenv("SPEAKER_PIPELINE", "pyannote/speaker-diarization-3.1")
    FORCED_NUM_SPEAKERS = int(os.getenv("NUM_SPEAKERS", "2"))

    # 1) WhisperX ASR + alignment
    try:
        import whisperx
    except Exception:
        whisperx = None

    aligned = None
    if use_whisperx and whisperx is not None:
        try:
            runtime_device = "cuda" if (device != "cpu" and __import__("torch").cuda.is_available()) else "cpu"
            model = whisperx.load_model(model_size, device=runtime_device)
            result = model.transcribe(audio_path)

            language = result.get("language", "en")
            align_model, metadata = whisperx.load_align_model(language_code=language, device=runtime_device)
            aligned = whisperx.align(result["segments"], align_model, metadata, audio_path, device=runtime_device)
        except Exception as e:
            print(f"[transcribe] whisperx failed: {e}")
            aligned = None

    # If WhisperX not available, fall back to ASR without alignment
    if aligned is None:
        try:
            from faster_whisper import WhisperModel
        except Exception:
            WhisperModel = None
        if WhisperModel is not None:
            compute = compute_type if compute_type is not None else ("float16" if device != "cpu" else "float32")
            try:
                model = WhisperModel(model_size, device=device, compute_type=compute)
                result = model.transcribe(audio_path, beam_size=5, vad_filter=False)
                segments_out = []
                for segment in result[0]:
                    segments_out.append({"start": float(segment.start), "end": float(segment.end), "speaker": "Unknown", "text": segment.text.strip()})
                return segments_out
            except Exception as e:
                raise RuntimeError(f"ASR fallback failed: {e}")
        raise RuntimeError("No transcription backend available (install whisperx or faster-whisper).")

    # Normalize word-level alignment
    word_segments = aligned.get("word_segments", None) or aligned.get("words", None) or []
    normalized_words = []
    for w in word_segments:
        if isinstance(w, dict) and "start" in w:
            normalized_words.append({
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", w.get("start", 0.0))),
                "word": w.get("word", "")
            })

    # 2) Pyannote: speech separation + diarization (preferred)
    diarization_segments = None
    try:
        from pyannote.audio import Pipeline
        sep_pipeline = Pipeline.from_pretrained(SEPARATION_PIPELINE, use_auth_token=hf_token)
        # Optional: push to GPU if available
        try:
            import torch as _torch
            if device != "cpu" and _torch.cuda.is_available():
                sep_pipeline.to(_torch.device("cuda"))
        except Exception:
            pass

        diarization, sources = sep_pipeline(audio_path)
        diarization_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarization_segments.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker)
            })
    except Exception as e:
        print(f"[transcribe] separation pipeline failed: {e}")

    # 2b) Fallback diarization (force 2 speakers) if separation yielded <=1 speaker
    if not diarization_segments or len({d["speaker"] for d in diarization_segments}) <= 1:
        try:
            from pyannote.audio import Pipeline as _P2
            diar_p = _P2.from_pretrained(FALLBACK_DIAR_PIPELINE, use_auth_token=hf_token)
            # Optional GPU
            try:
                import torch as _torch2
                if device != "cpu" and _torch2.cuda.is_available():
                    diar_p.to(_torch2.device("cuda"))
            except Exception:
                pass
            diar = diar_p(audio_path, num_speakers=FORCED_NUM_SPEAKERS)
            diarization_segments = []
            for turn, _, speaker in diar.itertracks(yield_label=True):
                diarization_segments.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker)
                })
        except Exception as e:
            print(f"[transcribe] fallback diarization failed: {e}")

    # 3) Map words -> speakers and group
    if diarization_segments and normalized_words:
        words_with_speakers = _map_words_to_speakers_by_diarization(normalized_words, diarization_segments)
        grouped = _group_words_to_segments(words_with_speakers, gap_threshold=0.9)
        # If still single speaker, at least preserve aligned segment boundaries
        if len({g["speaker"] for g in grouped}) <= 1:
            simple = []
            for seg in aligned.get("segments", []):
                simple.append({
                    "start": float(seg.get("start", 0.0)),
                    "end": float(seg.get("end", 0.0)),
                    "speaker": grouped[0]["speaker"] if grouped else "Unknown",
                    "text": seg.get("text", "").strip()
                })
            return simple
        return grouped

    # If diarization unavailable, return aligned segments with Unknown speaker
    out = []
    for seg in aligned.get("segments", []):
        out.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "speaker": "Unknown",
            "text": seg.get("text", "").strip()
        })
    return out
