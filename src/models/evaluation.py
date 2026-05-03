"""
Evaluation Module — Enhanced for Week 15
Provides WER calculation, latency benchmarking, per-component profiling,
quality scoring, and comprehensive comparison across settings and noise presets.
"""

import time
import logging
import numpy as np
from jiwer import wer, mer, wil

logger = logging.getLogger(__name__)


class TranscriptionEvaluator:
    """
    Evaluates speech transcription quality using standard ASR metrics.
    Supports baseline vs. improved setting comparisons, noise robustness
    matrix, and per-component latency profiling.
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
    def compute_cer(reference: str, hypothesis: str) -> float:
        """
        Compute Character Error Rate.

        Args:
            reference: Ground-truth text.
            hypothesis: Predicted text.

        Returns:
            CER as a float.
        """
        if not reference or not hypothesis:
            return None

        ref_chars = list(reference.lower().strip())
        hyp_chars = list(hypothesis.lower().strip())

        # Simple Levenshtein distance at character level
        n, m = len(ref_chars), len(hyp_chars)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref_chars[i - 1] == hyp_chars[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

        cer_val = dp[n][m] / max(n, 1)
        return round(cer_val, 4)

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

    @staticmethod
    def compute_quality_score(
        wer_val: float = None,
        latency_seconds: float = None,
        audio_duration: float = None,
        word_count: int = None,
    ) -> dict:
        """
        Compute an overall quality score (0–100) based on multiple factors.

        Args:
            wer_val: Word Error Rate (0.0–1.0+).
            latency_seconds: Transcription latency.
            audio_duration: Audio length in seconds.
            word_count: Number of words in transcript.

        Returns:
            dict with score, grade, and component scores.
        """
        scores = {}

        # WER score (0-40 points): lower WER = higher score
        if wer_val is not None:
            wer_score = max(0, 40 * (1 - min(wer_val, 1.0)))
            scores["accuracy"] = round(wer_score, 1)
        else:
            scores["accuracy"] = 25  # Default neutral score when no reference

        # Speed score (0-30 points): real-time factor
        if latency_seconds and audio_duration and audio_duration > 0:
            rtf = latency_seconds / audio_duration
            speed_score = max(0, 30 * (1 - min(rtf, 2.0) / 2))
            scores["speed"] = round(speed_score, 1)
        else:
            scores["speed"] = 15

        # Completeness score (0-30 points): words per second of audio
        if word_count and audio_duration and audio_duration > 0:
            wps = word_count / audio_duration
            # Typical speech: 2-3 words/sec
            if 1.0 <= wps <= 4.0:
                completeness = 30
            elif 0.5 <= wps < 1.0 or 4.0 < wps <= 5.0:
                completeness = 20
            else:
                completeness = 10
            scores["completeness"] = completeness
        else:
            scores["completeness"] = 15

        total = sum(scores.values())

        # Letter grade
        if total >= 90:
            grade = "A+"
        elif total >= 80:
            grade = "A"
        elif total >= 70:
            grade = "B"
        elif total >= 60:
            grade = "C"
        elif total >= 50:
            grade = "D"
        else:
            grade = "F"

        return {
            "total_score": round(total, 1),
            "grade": grade,
            "components": scores,
            "max_score": 100,
        }

    def profile_pipeline_latency(
        self,
        whisper_model,
        nlp_pipeline,
        audio_array: np.ndarray,
        sampling_rate: int = 16000,
        language: str = None,
    ) -> dict:
        """
        Profile latency of each pipeline component.

        Returns:
            dict with per-component timing breakdown.
        """
        timings = {}

        # ASR
        t0 = time.time()
        result = whisper_model.transcribe(
            audio_array, sampling_rate=sampling_rate, language=language
        )
        timings["asr"] = round(time.time() - t0, 3)
        transcript = result["text"]

        # Summarization
        t0 = time.time()
        if len(transcript.split()) > 20:
            nlp_pipeline.summarize(transcript)
        timings["summarization"] = round(time.time() - t0, 3)

        # Sentiment
        t0 = time.time()
        nlp_pipeline.analyze_sentiment(transcript)
        timings["sentiment"] = round(time.time() - t0, 3)

        # Action items
        t0 = time.time()
        nlp_pipeline.extract_action_items(transcript)
        timings["action_items"] = round(time.time() - t0, 3)

        # Topics
        t0 = time.time()
        nlp_pipeline.extract_key_topics(transcript)
        timings["topics"] = round(time.time() - t0, 3)

        total = sum(timings.values())
        timings["total"] = round(total, 3)

        # Percentages
        percentages = {}
        for k, v in timings.items():
            if k != "total":
                percentages[k] = round(v / total * 100, 1) if total > 0 else 0

        return {
            "timings_seconds": timings,
            "percentages": percentages,
            "transcript": transcript,
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
                entry["cer"] = self.compute_cer(reference_text, text)

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
                entry["cer"] = self.compute_cer(reference_text, r["text"])

            results[name] = entry

        return results

    def noise_robustness_matrix(
        self,
        whisper_model,
        audio_array: np.ndarray,
        noise_presets: dict,
        apply_noise_fn,
        sampling_rate: int = 16000,
        reference_text: str = None,
        language: str = None,
    ) -> dict:
        """
        Generate a noise robustness matrix comparing all presets.

        Args:
            whisper_model: WhisperASR instance.
            audio_array: Clean audio.
            noise_presets: Dict of noise preset configurations.
            apply_noise_fn: Function to apply noise.
            sampling_rate: Sample rate.
            reference_text: Optional ground truth.
            language: Language code.

        Returns:
            dict with results per noise preset.
        """
        matrix = {}

        for preset_name in noise_presets:
            noisy_audio = apply_noise_fn(audio_array, preset_name)
            result = whisper_model.transcribe(
                noisy_audio, sampling_rate=sampling_rate, language=language
            )

            entry = {
                "text": result["text"],
                "latency_seconds": result["latency_seconds"],
                "word_count": len(result["text"].split()),
            }

            if reference_text:
                entry["wer_metrics"] = self.compute_wer(reference_text, result["text"])
                entry["cer"] = self.compute_cer(reference_text, result["text"])

            matrix[preset_name] = entry

        return matrix
