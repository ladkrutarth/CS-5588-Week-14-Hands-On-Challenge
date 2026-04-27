"""
CLI pipeline for local GPU Whisper-small project.

Usage examples:
python run_whisper_pipeline.py --audio samples/meeting.wav --task transcribe
python run_whisper_pipeline.py --audio samples/meeting.wav --reference-text "..." --run-eval
"""

import argparse
import json
from pathlib import Path

from models.evaluation import TranscriptionEvaluator
from models.nlp_pipeline import NLPPipeline
from models.whisper_model import WhisperASR
from utils.audio_utils import load_audio
from utils.noise_augmentation import apply_noise_preset


LANGUAGE_CHOICES = {
    "auto": None,
    "en": "en",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "zh": "zh",
    "ja": "ja",
    "ar": "ar",
    "hi": "hi",
    "pt": "pt",
    "ru": "ru",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Whisper-small speech intelligence pipeline.")
    parser.add_argument("--audio", required=True, help="Path to audio file.")
    parser.add_argument("--task", choices=["transcribe", "translate"], default="transcribe")
    parser.add_argument("--language", choices=list(LANGUAGE_CHOICES.keys()), default="auto")
    parser.add_argument("--noise-preset", default="Clean (No Noise)")
    parser.add_argument("--require-gpu", action="store_true", help="Fail if CUDA/MPS is unavailable.")
    parser.add_argument("--run-eval", action="store_true", help="Run baseline vs improved comparison.")
    parser.add_argument("--reference-text", default="", help="Ground-truth text for WER.")
    parser.add_argument("--output-json", default="outputs/latest_result.json")
    return parser.parse_args()


def ensure_gpu(device: str):
    if device == "cpu":
        raise RuntimeError(
            "GPU is required for this project, but model is on CPU. "
            "Install GPU PyTorch and run on CUDA (NVIDIA) or MPS (Apple Silicon)."
        )


def main():
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sr = load_audio(str(audio_path))
    processed_audio = apply_noise_preset(audio, args.noise_preset)

    whisper = WhisperASR()
    whisper.load_model()

    if args.require_gpu:
        ensure_gpu(whisper.device)

    language = LANGUAGE_CHOICES[args.language]
    result = whisper.transcribe(
        processed_audio,
        sampling_rate=sr,
        language=language,
        task=args.task,
    )

    nlp = NLPPipeline(device=whisper.device)
    transcript = result["text"]
    output = {
        "input": {
            "audio": str(audio_path),
            "task": args.task,
            "language": args.language,
            "noise_preset": args.noise_preset,
        },
        "asr": result,
        "analysis": {
            "summary": nlp.summarize(transcript) if len(transcript.split()) > 20 else transcript,
            "action_items": nlp.extract_action_items(transcript),
            "sentiment": nlp.analyze_sentiment(transcript),
            "topics": nlp.extract_key_topics(transcript),
        },
    }

    if args.run_eval:
        evaluator = TranscriptionEvaluator()
        eval_result = evaluator.quick_evaluate(
            whisper,
            processed_audio,
            sampling_rate=sr,
            reference_text=args.reference_text.strip() or None,
            language=language,
        )
        output["evaluation"] = eval_result

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"Device: {result['device']}")
    print(f"Latency (s): {result['latency_seconds']}")
    print(f"Transcript length (words): {len(transcript.split())}")
    print(f"Saved output JSON: {output_path}")


if __name__ == "__main__":
    main()
