"""
Local Text-to-Speech engine utilities.
Uses pyttsx3 for offline speech synthesis.
"""

from __future__ import annotations

import os
import re
import tempfile
import io
import time
import sys
import subprocess
from dataclasses import dataclass
import numpy as np
import soundfile as sf


@dataclass
class VoiceOption:
    id: str
    name: str
    gender: str = "unknown"
    locale: str = "unknown"
    language: str = "unknown"


class LocalTTSEngine:
    FEMALE_HINTS = {
        "samantha", "victoria", "karen", "moira", "susan", "ava", "allison", "siri",
        "fiona", "veena", "rishi", "tessa", "joana", "lisa", "monica", "anna",
    }
    MALE_HINTS = {
        "alex", "daniel", "fred", "jorge", "ralph", "diego", "thomas", "arnold",
        "serena", "aaron", "kanya", "nicky", "tom", "xander", "reed",
    }
    LANGUAGE_DEFAULT_VOICE = {
        "hi": "Lekha",
        "ja": "Kyoko",
        "zh": "Tingting",
        "ar": "Majed",
        "es": "Monica",
        "fr": "Thomas",
        "de": "Anna",
        "pt": "Luciana",
        "ru": "Milena",
        "en": "Samantha",
    }

    @staticmethod
    def _guess_gender(name: str) -> str:
        n = (name or "").strip().lower()
        if n in LocalTTSEngine.FEMALE_HINTS:
            return "female"
        if n in LocalTTSEngine.MALE_HINTS:
            return "male"
        return "unknown"

    def __init__(self):
        self._engine = None

    def _ensure_engine(self):
        if self._engine is None:
            try:
                import pyttsx3
            except ImportError as exc:
                raise RuntimeError(
                    "pyttsx3 is not installed. Run `pip install pyttsx3`."
                ) from exc
            self._engine = pyttsx3.init()
        return self._engine

    @staticmethod
    def sanitize_text(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        return cleaned

    def list_voices(self) -> list[VoiceOption]:
        if sys.platform == "darwin":
            try:
                output = subprocess.check_output(["say", "-v", "?"], text=True)
                voices = []
                for line in output.splitlines():
                    if not line.strip():
                        continue
                    # Typical line format: "Agnes               en_US    # sample sentence"
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    name = parts[0]
                    locale = parts[1]
                    language = locale.split("_")[0].lower() if "_" in locale else locale.lower()
                    voices.append(
                        VoiceOption(
                            id=name,
                            name=name,
                            gender=self._guess_gender(name),
                            locale=locale,
                            language=language,
                        )
                    )
                if voices:
                    return voices
            except Exception:
                pass

        engine = self._ensure_engine()
        voices = engine.getProperty("voices") or []
        result = []
        for voice in voices:
            voice_id = getattr(voice, "id", "")
            voice_name = getattr(voice, "name", voice_id)
            result.append(
                VoiceOption(
                    id=voice_id,
                    name=voice_name,
                    gender=self._guess_gender(voice_name),
                    locale="unknown",
                    language="unknown",
                )
            )
        return result

    def get_professional_voice_options(self, gender: str | None = None, language: str | None = None) -> list[VoiceOption]:
        voices = self.list_voices()
        if language and language != "auto":
            lang_norm = language.lower()
            lang_filtered = [v for v in voices if v.language == lang_norm]
            if lang_filtered:
                voices = lang_filtered
        if gender in {"male", "female"}:
            preferred = [v for v in voices if v.gender == gender]
            if preferred:
                return preferred
        return voices

    def synthesize_to_wav_bytes(
        self,
        text: str,
        rate: int = 170,
        voice_id: str | None = None,
        language: str | None = None,
    ) -> bytes:
        clean_text = self.sanitize_text(text)
        if not clean_text:
            raise ValueError("Text is empty after sanitization.")

        # Keep local TTS responsive for UI usage.
        if len(clean_text) > 2000:
            clean_text = clean_text[:2000]

        if sys.platform == "darwin":
            try:
                return self._synthesize_macos_wav(clean_text, voice_id=voice_id, language=language)
            except Exception:
                # Fallback to pyttsx3 backend if `say` path fails in this environment.
                return self._synthesize_pyttsx3_wav(clean_text, rate=rate, voice_id=voice_id)

        return self._synthesize_pyttsx3_wav(clean_text, rate=rate, voice_id=voice_id)

    def _synthesize_macos_wav(self, text: str, voice_id: str | None = None, language: str | None = None) -> bytes:
        """
        macOS robust synthesis:
        1) Try selected voice
        2) Retry default voice
        3) Split long text into chunks and stitch audio
        """
        chunks = self._chunk_for_tts(text, max_chars=280)
        preferred = self.LANGUAGE_DEFAULT_VOICE.get((language or "").lower())
        candidate_voices = [voice_id, preferred, None, "Samantha", "Alex"]
        candidate_voices = [v for i, v in enumerate(candidate_voices) if v not in candidate_voices[:i]]

        for candidate in candidate_voices:
            rendered = []
            sr_ref = None
            ok = True
            for chunk in chunks:
                data, sr = self._render_macos_chunk(chunk, candidate)
                if data is None or getattr(data, "size", 0) == 0:
                    ok = False
                    break
                sr_ref = sr if sr_ref is None else sr_ref
                rendered.append(data)
            if ok and rendered:
                merged = np.concatenate(rendered, axis=0)
                out = io.BytesIO()
                sf.write(out, merged, sr_ref or 22050, format="WAV", subtype="PCM_16")
                return out.getvalue()

        raise RuntimeError("TTS engine produced empty audio. Try a different voice or shorter text.")

    @staticmethod
    def _chunk_for_tts(text: str, max_chars: int = 280) -> list[str]:
        text = text.strip()
        if len(text) <= max_chars:
            return [text]
        parts = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""
        for part in parts:
            if not part:
                continue
            if len(current) + len(part) + 1 <= max_chars:
                current = f"{current} {part}".strip()
            else:
                if current:
                    chunks.append(current)
                current = part
        if current:
            chunks.append(current)
        return chunks if chunks else [text[:max_chars]]

    def _render_macos_chunk(self, chunk: str, voice_id: str | None):
        fd, path = tempfile.mkstemp(suffix=".aiff")
        os.close(fd)
        try:
            cmd = ["say", "-o", path]
            if voice_id:
                cmd.extend(["-v", voice_id])
            cmd.append(chunk)
            subprocess.run(cmd, check=True)
            self._wait_for_file(path, timeout_seconds=3.0)
            data, sr = sf.read(path, dtype="float32")
            return data, sr
        except Exception:
            return None, None
        finally:
            if os.path.exists(path):
                os.remove(path)

    @staticmethod
    def _wait_for_file(path: str, timeout_seconds: float = 5.0):
        deadline = time.time() + timeout_seconds
        last_size = -1
        stable_reads = 0
        while time.time() < deadline:
            size = os.path.getsize(path) if os.path.exists(path) else 0
            if size > 0 and size == last_size:
                stable_reads += 1
                if stable_reads >= 2:
                    return
            else:
                stable_reads = 0
            last_size = size
            time.sleep(0.1)

    def _synthesize_pyttsx3_wav(self, clean_text: str, rate: int, voice_id: str | None) -> bytes:
        suffix = ".wav"
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            engine = self._ensure_engine()
            engine.setProperty("rate", int(rate))
            if voice_id:
                try:
                    engine.setProperty("voice", voice_id)
                except Exception:
                    pass
            engine.save_to_file(clean_text, path)
            engine.runAndWait()
            self._wait_for_file(path, timeout_seconds=5.0)

            data, sr = sf.read(path, dtype="float32")
            if getattr(data, "size", 0) == 0:
                raise RuntimeError("TTS engine produced empty audio. Try a different voice or shorter text.")
            output = io.BytesIO()
            sf.write(output, data, sr, format="WAV", subtype="PCM_16")
            return output.getvalue()
        finally:
            if os.path.exists(path):
                os.remove(path)
