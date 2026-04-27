"""
Audio Processing Utilities
Handles loading, resampling, format conversion, and audio info extraction.
"""

import io
import numpy as np
import librosa
import soundfile as sf


def load_audio(file_path_or_bytes, target_sr: int = 16000) -> tuple:
    """
    Load audio from a file path or bytes object and resample to target rate.

    Args:
        file_path_or_bytes: Path string or file-like bytes object.
        target_sr: Target sampling rate (Whisper requires 16kHz).

    Returns:
        Tuple of (audio_array, sampling_rate).
    """
    if isinstance(file_path_or_bytes, (str,)):
        audio, sr = librosa.load(file_path_or_bytes, sr=target_sr, mono=True)
    else:
        audio, sr = sf.read(io.BytesIO(file_path_or_bytes))
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

    return audio.astype(np.float32), sr


def get_audio_info(audio_array: np.ndarray, sampling_rate: int) -> dict:
    """Extract metadata about an audio signal."""
    duration = len(audio_array) / sampling_rate
    rms = float(np.sqrt(np.mean(audio_array ** 2)))
    peak = float(np.max(np.abs(audio_array)))

    return {
        "duration_seconds": round(duration, 2),
        "sampling_rate": sampling_rate,
        "num_samples": len(audio_array),
        "rms_energy": round(rms, 4),
        "peak_amplitude": round(peak, 4),
        "is_silent": rms < 0.001,
    }


def trim_silence(audio_array: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Trim leading and trailing silence from audio."""
    trimmed, _ = librosa.effects.trim(audio_array, top_db=top_db)
    return trimmed
