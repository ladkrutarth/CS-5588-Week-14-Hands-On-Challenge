"""
Noise Augmentation Utilities
Add various types of noise to clean audio for evaluation testing.
Compares clean vs noisy transcription quality.
"""

import numpy as np


def add_white_noise(audio: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add white Gaussian noise at a specified SNR level."""
    rms_signal = np.sqrt(np.mean(audio ** 2))
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, audio.shape).astype(np.float32)
    return audio + noise


def add_ambient_noise(audio: np.ndarray, snr_db: float = 15.0) -> np.ndarray:
    """Simulate ambient/background noise using filtered noise."""
    rms_signal = np.sqrt(np.mean(audio ** 2))
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    # Brown noise approximation (low-freq dominant)
    white = np.random.normal(0, 1, audio.shape)
    brown = np.cumsum(white)
    brown = brown / np.max(np.abs(brown))
    brown = (brown * rms_noise).astype(np.float32)
    return audio + brown


def add_reverb_effect(audio: np.ndarray, decay: float = 0.3, delay_ms: int = 50, sr: int = 16000) -> np.ndarray:
    """Add simple reverb effect."""
    delay_samples = int(sr * delay_ms / 1000)
    output = audio.copy()
    if delay_samples < len(audio):
        output[delay_samples:] += decay * audio[:-delay_samples]
    # Normalize to prevent clipping
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / max_val
    return output


NOISE_PRESETS = {
    "Clean (No Noise)": {"func": None, "params": {}},
    "Light Noise (SNR 30dB)": {"func": add_white_noise, "params": {"snr_db": 30.0}},
    "Moderate Noise (SNR 20dB)": {"func": add_white_noise, "params": {"snr_db": 20.0}},
    "Heavy Noise (SNR 10dB)": {"func": add_white_noise, "params": {"snr_db": 10.0}},
    "Ambient Noise (SNR 15dB)": {"func": add_ambient_noise, "params": {"snr_db": 15.0}},
    "Reverb Effect": {"func": add_reverb_effect, "params": {"decay": 0.4}},
}


def apply_noise_preset(audio: np.ndarray, preset_name: str) -> np.ndarray:
    """Apply a named noise preset to audio."""
    preset = NOISE_PRESETS.get(preset_name)
    if preset is None or preset["func"] is None:
        return audio
    return preset["func"](audio, **preset["params"])
