"""
Audio Processing Utilities — Enhanced for Week 15
Handles loading, resampling, format conversion, normalization,
silence trimming, bandpass filtering, and audio info extraction.
"""

import io
import logging
import numpy as np
import librosa
import soundfile as sf
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)


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
    """Extract comprehensive metadata about an audio signal."""
    duration = len(audio_array) / sampling_rate
    rms = float(np.sqrt(np.mean(audio_array ** 2)))
    peak = float(np.max(np.abs(audio_array)))

    # Compute signal-to-noise ratio estimate
    snr_estimate = _estimate_snr(audio_array, sampling_rate)

    # Compute spectral centroid for audio characterization
    try:
        spectral_centroid = float(np.mean(
            librosa.feature.spectral_centroid(y=audio_array, sr=sampling_rate)
        ))
    except Exception:
        spectral_centroid = 0.0

    return {
        "duration_seconds": round(duration, 2),
        "sampling_rate": sampling_rate,
        "num_samples": len(audio_array),
        "rms_energy": round(rms, 4),
        "peak_amplitude": round(peak, 4),
        "is_silent": rms < 0.001,
        "snr_estimate_db": round(snr_estimate, 1),
        "spectral_centroid_hz": round(spectral_centroid, 1),
    }


def trim_silence(audio_array: np.ndarray, top_db: int = 30) -> np.ndarray:
    """Trim leading and trailing silence from audio."""
    trimmed, _ = librosa.effects.trim(audio_array, top_db=top_db)
    return trimmed


def normalize_audio(audio_array: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to a target loudness level.

    Args:
        audio_array: Input audio waveform.
        target_db: Target RMS loudness in dB.

    Returns:
        Normalized audio array.
    """
    rms = np.sqrt(np.mean(audio_array ** 2))
    if rms < 1e-8:
        return audio_array  # Silence, nothing to normalize

    target_rms = 10 ** (target_db / 20)
    gain = target_rms / rms
    normalized = audio_array * gain

    # Prevent clipping
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        normalized = normalized / peak * 0.99

    logger.info(f"Audio normalized: gain={gain:.2f}x, target={target_db}dB")
    return normalized.astype(np.float32)


def apply_bandpass_filter(
    audio_array: np.ndarray,
    sr: int = 16000,
    low_freq: float = 80.0,
    high_freq: float = 7500.0,
    order: int = 5,
) -> np.ndarray:
    """
    Apply a bandpass filter to focus on speech frequencies.

    Args:
        audio_array: Input audio waveform.
        sr: Sampling rate.
        low_freq: Lower cutoff frequency (Hz).
        high_freq: Upper cutoff frequency (Hz).
        order: Filter order.

    Returns:
        Filtered audio array.
    """
    nyquist = sr / 2
    low = low_freq / nyquist
    high = min(high_freq / nyquist, 0.99)

    b, a = scipy_signal.butter(order, [low, high], btype="band")
    filtered = scipy_signal.filtfilt(b, a, audio_array)

    logger.info(f"Bandpass filter applied: {low_freq}Hz–{high_freq}Hz")
    return filtered.astype(np.float32)


def preprocess_audio(
    audio_array: np.ndarray,
    sr: int = 16000,
    trim: bool = True,
    normalize: bool = True,
    bandpass: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for optimal Whisper input.

    Steps:
    1. Trim leading/trailing silence
    2. Normalize loudness
    3. Apply speech-band filter

    Args:
        audio_array: Raw audio waveform.
        sr: Sampling rate.
        trim: Whether to trim silence.
        normalize: Whether to normalize loudness.
        bandpass: Whether to apply bandpass filter.

    Returns:
        Preprocessed audio array.
    """
    result = audio_array.copy()

    if trim:
        result = trim_silence(result)
        logger.info(f"Trimmed silence: {len(audio_array)} -> {len(result)} samples")

    if normalize:
        result = normalize_audio(result)

    if bandpass:
        result = apply_bandpass_filter(result, sr=sr)

    return result


def _estimate_snr(audio_array: np.ndarray, sr: int) -> float:
    """
    Estimate signal-to-noise ratio using a simple energy-based method.
    Splits audio into frames and compares high-energy (signal) vs low-energy (noise) frames.
    """
    frame_length = int(sr * 0.025)  # 25ms frames
    hop_length = int(sr * 0.010)    # 10ms hop

    frames = librosa.util.frame(audio_array, frame_length=frame_length, hop_length=hop_length)
    frame_energies = np.mean(frames ** 2, axis=0)

    if len(frame_energies) == 0:
        return 0.0

    # Sort energies and use bottom 20% as noise estimate
    sorted_energies = np.sort(frame_energies)
    noise_frames = max(1, len(sorted_energies) // 5)
    noise_energy = np.mean(sorted_energies[:noise_frames])
    signal_energy = np.mean(sorted_energies[noise_frames:])

    if noise_energy < 1e-10:
        return 60.0  # Very clean audio

    snr = 10 * np.log10(signal_energy / noise_energy)
    return float(snr)


def compute_waveform_data(audio_array: np.ndarray, sr: int, num_points: int = 500) -> dict:
    """
    Compute downsampled waveform data for visualization.

    Args:
        audio_array: Audio waveform.
        sr: Sampling rate.
        num_points: Number of display points.

    Returns:
        dict with time and amplitude arrays.
    """
    duration = len(audio_array) / sr
    step = max(1, len(audio_array) // num_points)

    # Compute envelope
    amplitudes = []
    times = []
    for i in range(0, len(audio_array) - step, step):
        chunk = audio_array[i:i + step]
        amplitudes.append(float(np.max(np.abs(chunk))))
        times.append(i / sr)

    return {
        "times": times,
        "amplitudes": amplitudes,
        "duration": duration,
        "num_points": len(times),
    }
