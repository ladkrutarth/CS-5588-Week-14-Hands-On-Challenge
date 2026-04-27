"""
Evaluation Module
Provides WER calculation, latency benchmarking, and comparison
between baseline and improved transcription settings.
"""

import time
import numpy as np
from jiwer import wer, mer, wil


class TranscriptionEvaluator:
    """
    Evaluates speech transcription quality using standard ASR metrics.
    Supports baseline vs. improved setting comparisons.
    """

    @staticmethod
    def compute_wer(reference: str, hypothesis: str) -> dict:
        """
        Compute Word Error Rate and related metrics.

        Args:
            reference: Ground-truth transcription.
            hypothesis: Model-generated transcription.

        Returns:
            dict with WER, MER, WIL metrics.
        """
        if not reference or not hypothesis:
            return {"wer": None, "mer": None, "wil": None, "error": "Empty input"}

        ref = reference.lower().strip()
        hyp = hypothesis.lower().strip()

        return {
            "wer": round(wer(ref, hyp), 4),
            "mer": round(mer(ref, hyp), 4),
            "wil": round(wil(ref, hyp), 4),
        }

    @staticmethod
    def compute_latency_stats(latencies: list) -> dict:
        """
        Compute latency statistics from a list of measurements.

        Args:
            latencies: List of latency values in seconds.

        Returns:
            dict with mean, median, std, min, max.
        """
        if not latencies:
            return {}

        arr = np.array(latencies)
        return {
            "mean_seconds": round(float(np.mean(arr)), 4),
            "median_seconds": round(float(np.median(arr)), 4),
            "std_seconds": round(float(np.std(arr)), 4),
            "min_seconds": round(float(np.min(arr)), 4),
            "max_seconds": round(float(np.max(arr)), 4),
            "num_samples": len(latencies),
        }

    def compare_settings(
        self,
        whisper_model,
        audio_array: np.ndarray,
        sampling_rate: int = 16000,
        reference_text: str = None,
        language: str = None,
    ) -> dict:
        """
        Compare baseline (greedy) vs. improved (beam search) transcription.

        Args:
            whisper_model: WhisperASR instance.
            audio_array: Audio waveform.
            sampling_rate: Sample rate.
            reference_text: Optional ground-truth for WER calculation.
            language: Language code.

        Returns:
            dict with results for each configuration.
        """
        configs = {
            "Baseline (Greedy)": {"beam_size": 1, "temperature": 0.0},
            "Beam Search (5)": {"beam_size": 5, "temperature": 0.0},
            "Beam Search (10)": {"beam_size": 10, "temperature": 0.0},
            "Sampling (temp=0.3)": {"beam_size": 1, "temperature": 0.3},
            "Sampling (temp=0.7)": {"beam_size": 1, "temperature": 0.7},
        }

        results = {}

        for name, cfg in configs.items():
            # Run multiple times for stable latency
            latencies = []
            text = ""

            for run in range(3):
                r = whisper_model.transcribe_with_settings(
                    audio_array,
                    sampling_rate=sampling_rate,
                    language=language,
                    beam_size=cfg["beam_size"],
                    temperature=cfg["temperature"],
                )
                latencies.append(r["latency_seconds"])
                text = r["text"]

            entry = {
                "text": text,
                "latency": self.compute_latency_stats(latencies),
                "settings": cfg,
            }

            if reference_text:
                entry["wer_metrics"] = self.compute_wer(reference_text, text)

            results[name] = entry

        return results

    def quick_evaluate(
        self,
        whisper_model,
        audio_array: np.ndarray,
        sampling_rate: int = 16000,
        reference_text: str = None,
        language: str = None,
    ) -> dict:
        """
        Quick evaluation: baseline vs beam search (2 settings only).
        Faster than compare_settings().
        """
        configs = {
            "Baseline (Greedy, beam=1)": {"beam_size": 1, "temperature": 0.0},
            "Improved (Beam Search, beam=5)": {"beam_size": 5, "temperature": 0.0},
        }

        results = {}
        for name, cfg in configs.items():
            r = whisper_model.transcribe_with_settings(
                audio_array,
                sampling_rate=sampling_rate,
                language=language,
                beam_size=cfg["beam_size"],
                temperature=cfg["temperature"],
            )

            entry = {
                "text": r["text"],
                "latency_seconds": r["latency_seconds"],
                "settings": cfg,
            }

            if reference_text:
                entry["wer_metrics"] = self.compute_wer(reference_text, r["text"])

            results[name] = entry

        return results
