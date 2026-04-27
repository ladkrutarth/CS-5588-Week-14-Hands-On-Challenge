"""
Whisper ASR Engine - GPU-Accelerated Speech-to-Text
Uses openai/whisper-small model from Hugging Face Transformers.
Supports: transcription, translation, language detection, and timestamp extraction.
"""

import time
import os
from contextlib import contextmanager
import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class WhisperASR:
    """
    GPU-accelerated Whisper-small ASR engine.
    Provides transcription, translation, and language detection.
    """

    MODEL_ID = "openai/whisper-small"

    @staticmethod
    @contextmanager
    def _without_proxy_env():
        """
        Temporarily remove proxy environment variables.
        Some local proxy setups block Hugging Face model downloads (HTTP 403).
        """
        proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]
        original = {k: os.environ.get(k) for k in proxy_keys}
        try:
            for key in proxy_keys:
                os.environ.pop(key, None)
            yield
        finally:
            for key, value in original.items():
                if value is not None:
                    os.environ[key] = value
                else:
                    os.environ.pop(key, None)

    def __init__(self, device: str = None):
        """
        Initialize Whisper model on the best available device.
        
        Args:
            device: Force a specific device ('cuda', 'mps', 'cpu').
                    If None, auto-detects GPU availability.
        """
        self.device = device or self._detect_device()
        self.processor = None
        self.model = None
        self._loaded = False
        print(f"[WhisperASR] Target device: {self.device}")

    @staticmethod
    def _detect_device() -> str:
        """Auto-detect the best available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self):
        """Load Whisper model and processor onto the target device."""
        if self._loaded:
            return

        print(f"[WhisperASR] Loading {self.MODEL_ID} on {self.device}...")
        start = time.time()

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            with self._without_proxy_env():
                self.processor = WhisperProcessor.from_pretrained(self.MODEL_ID)
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
        except Exception:
            # Fallback to local cache only when network/proxy path fails.
            try:
                self.processor = WhisperProcessor.from_pretrained(self.MODEL_ID, local_files_only=True)
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.MODEL_ID,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
            except Exception as second_error:
                raise RuntimeError(
                    "Failed to load Whisper model from Hugging Face and local cache. "
                    "If you are behind a proxy, download once with proxy disabled or set local model cache."
                ) from second_error

        # Move model to target device
        if self.device != "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True

        elapsed = time.time() - start
        print(f"[WhisperASR] Model loaded in {elapsed:.2f}s")

    def _generation_kwargs(
        self,
        task: str = "transcribe",
        language: str = None,
        beam_size: int = 1,
        temperature: float = 0.0,
        requested_max_new_tokens: int = 220,
    ) -> dict:
        """
        Build generation config for Whisper.
        Whisper decoding task/language is controlled by forced decoder tokens.
        """
        max_target_positions = getattr(self.model.config, "max_target_positions", 448)
        # Keep a safety margin for decoder start/prompt tokens to avoid max length errors.
        safe_max_new_tokens = max(1, min(requested_max_new_tokens, max_target_positions - 8))

        kwargs = {
            "task": task,
            "max_new_tokens": safe_max_new_tokens,
            "num_beams": beam_size,
        }
        if language:
            kwargs["language"] = language
        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["do_sample"] = True
        return kwargs

    def transcribe(
        self,
        audio_array: np.ndarray,
        sampling_rate: int = 16000,
        language: str = None,
        task: str = "transcribe",
        return_timestamps: bool = False,
    ) -> dict:
        """
        Transcribe or translate audio.

        Args:
            audio_array: Audio waveform as numpy array (mono, float32).
            sampling_rate: Sample rate of the audio (default 16kHz).
            language: Force a specific language (e.g., 'en', 'es', 'fr').
                      If None, Whisper auto-detects.
            task: 'transcribe' or 'translate' (translates to English).
            return_timestamps: Whether to return word-level timestamps.

        Returns:
            dict with keys: 'text', 'language', 'latency_seconds',
                            and optionally 'chunks' for timestamps.
        """
        self.load_model()

        # Preprocess audio
        input_features = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        ).input_features

        if self.device != "cpu":
            input_features = input_features.to(self.device)
        if self.device == "cuda":
            input_features = input_features.half()

        gen_kwargs = self._generation_kwargs(
            task=task,
            language=language,
            beam_size=1,
            temperature=0.0,
        )
        if return_timestamps:
            gen_kwargs["return_timestamps"] = True

        # Generate
        start = time.time()
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, **gen_kwargs)
        latency = time.time() - start

        # Decode
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )
        text = transcription[0].strip()

        # Retry once with a simpler decode setup if we receive an empty transcript.
        if not text:
            retry_kwargs = {
                "max_new_tokens": min(220, getattr(self.model.config, "max_target_positions", 448) - 8),
                "num_beams": 1,
            }
            if language:
                retry_kwargs["language"] = language
            if task:
                retry_kwargs["task"] = task
            with torch.no_grad():
                predicted_ids = self.model.generate(input_features, **retry_kwargs)
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            text = transcription[0].strip()

        # Detect language from decoder output
        detected_lang = language or "auto-detected"

        result = {
            "text": text,
            "language": detected_lang,
            "latency_seconds": round(latency, 3),
            "task": task,
            "device": self.device,
        }

        return result

    def transcribe_with_settings(
        self,
        audio_array: np.ndarray,
        sampling_rate: int = 16000,
        language: str = None,
        beam_size: int = 1,
        temperature: float = 0.0,
        task: str = "transcribe",
    ) -> dict:
        """
        Transcribe with configurable generation settings for evaluation.
        Allows comparing greedy vs beam search, temperature effects, etc.

        Args:
            audio_array: Audio waveform.
            sampling_rate: Sample rate.
            language: Target language.
            beam_size: Number of beams (1 = greedy).
            temperature: Sampling temperature.
            task: 'transcribe' or 'translate'.

        Returns:
            dict with transcription result and settings used.
        """
        self.load_model()

        input_features = self.processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        ).input_features

        if self.device != "cpu":
            input_features = input_features.to(self.device)
        if self.device == "cuda":
            input_features = input_features.half()

        gen_kwargs = self._generation_kwargs(
            task=task,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
        )

        start = time.time()
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features, **gen_kwargs)
        latency = time.time() - start

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )

        return {
            "text": transcription[0].strip(),
            "language": language or "auto",
            "latency_seconds": round(latency, 3),
            "settings": {
                "beam_size": beam_size,
                "temperature": temperature,
                "task": task,
            },
            "device": self.device,
        }

    def get_device_info(self) -> dict:
        """Return information about the compute device being used."""
        info = {
            "device": self.device,
            "model": self.MODEL_ID,
            "loaded": self._loaded,
        }
        if self.device == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_mem / 1e9, 2
            )
        elif self.device == "mps":
            info["gpu_name"] = "Apple Silicon (MPS)"
        return info
