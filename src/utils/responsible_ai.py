"""
Responsible AI Module
Provides bias detection, content sensitivity scanning, consent framework,
and AI transparency reporting for the Speech Intelligence Platform.
"""

import re
import logging
from datetime import datetime
from collections import Counter

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Sensitive Information Detection
# ------------------------------------------------------------------ #
SENSITIVE_PATTERNS = {
    "SSN": r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
    "Credit Card": r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
    "Email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "Phone": r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "IP Address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
}


def scan_sensitive_content(text: str) -> dict:
    """
    Scan text for potentially sensitive/private information.

    Args:
        text: Input text to scan.

    Returns:
        dict with detected sensitive items and risk level.
    """
    findings = {}
    for name, pattern in SENSITIVE_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            # Partially mask the findings for privacy
            masked = [_mask_value(m) for m in matches]
            findings[name] = {
                "count": len(matches),
                "examples": masked[:3],  # Show at most 3
            }

    risk_level = "low"
    if findings:
        total = sum(f["count"] for f in findings.values())
        if total >= 5:
            risk_level = "high"
        elif total >= 2:
            risk_level = "medium"
        else:
            risk_level = "low"

    return {
        "findings": findings,
        "risk_level": risk_level,
        "has_sensitive_content": len(findings) > 0,
        "scan_timestamp": datetime.now().isoformat(),
    }


def _mask_value(value: str) -> str:
    """Partially mask a sensitive value for display."""
    if len(value) <= 4:
        return "****"
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


# ------------------------------------------------------------------ #
#  Bias Detection
# ------------------------------------------------------------------ #
def detect_transcription_bias(
    transcript: str,
    audio_duration_seconds: float = 0.0,
    language: str | None = None,
) -> dict:
    """
    Detect potential bias or quality issues in transcription output.

    Checks for:
    - Suspiciously short transcripts relative to audio length
    - High word repetition (possible hallucination)
    - Common Whisper hallucination patterns

    Args:
        transcript: The transcribed text.
        audio_duration_seconds: Duration of the source audio.

    Returns:
        dict with bias warnings and recommendations.
    """
    warnings = []
    recommendations = []

    words = transcript.split()
    word_count = len(words)

    # Check 1: Transcript too short for audio length
    if audio_duration_seconds > 10 and word_count < 5:
        warnings.append({
            "type": "short_transcript",
            "severity": "high",
            "message": f"Only {word_count} words for {audio_duration_seconds:.0f}s audio — possible transcription failure.",
        })
        recommendations.append("Try a different language setting or check audio quality.")

    # Check 2: High repetition ratio (hallucination indicator)
    if word_count > 10:
        # Normalize tokens and skip common filler/grammar words per language to reduce false positives.
        lang = (language or "").lower()
        stopwords = {
            "generic": {"the", "a", "an", "and", "is", "are", "to", "of", "in", "on", "for", "it", "that"},
            "hi": {"है", "हैं", "और", "के", "की", "का", "को", "में", "से", "पर", "यह", "वह"},
            "en": {"the", "is", "are", "and", "to", "of", "in", "it", "that"},
        }
        lang_stop = stopwords.get(lang, set()) | stopwords["generic"]

        cleaned = []
        for w in words:
            t = re.sub(r"[^\w\u0900-\u097F]+", "", w.lower()).strip()
            if not t:
                continue
            if t in lang_stop:
                continue
            if len(t) < 2:
                continue
            cleaned.append(t)

        if cleaned:
            word_freq = Counter(cleaned)
            top_word, top_count = word_freq.most_common(1)[0]
            repetition_ratio = top_count / len(cleaned)
        else:
            top_word, top_count, repetition_ratio = "", 0, 0.0

        # Trigger only if repetition is both dominant and frequent.
        if repetition_ratio > 0.45 and top_count >= 4:
            warnings.append({
                "type": "high_repetition",
                "severity": "medium",
                "message": (
                    f"High word repetition detected ({repetition_ratio:.0%}) "
                    f"for token '{top_word}'. Possible model hallucination."
                ),
            })
            recommendations.append("Try different beam size or noise reduction settings.")

    # Check 3: Common Whisper hallucination patterns
    hallucination_patterns = [
        r"thank you for watching",
        r"please subscribe",
        r"like and subscribe",
        r"see you in the next",
        r"(?:\.\.\.){3,}",  # Repeated dots
    ]
    for pattern in hallucination_patterns:
        if re.search(pattern, transcript, re.IGNORECASE):
            warnings.append({
                "type": "possible_hallucination",
                "severity": "medium",
                "message": "Detected common Whisper hallucination pattern in transcript.",
            })
            recommendations.append("Verify this text was actually spoken in the audio.")
            break

    # Check 4: Empty or near-empty output
    if not transcript.strip():
        warnings.append({
            "type": "empty_transcript",
            "severity": "high",
            "message": "Transcript is empty. Audio may be silence, non-speech, or in an unsupported language.",
        })
        recommendations.append("Check that the audio contains clear speech.")

    return {
        "warnings": warnings,
        "recommendations": recommendations,
        "warning_count": len(warnings),
        "has_issues": len(warnings) > 0,
        "check_timestamp": datetime.now().isoformat(),
    }


# ------------------------------------------------------------------ #
#  Voice Cloning Consent Framework
# ------------------------------------------------------------------ #
CONSENT_TERMS = {
    "title": "Voice Synthesis Consent & Usage Policy",
    "version": "1.0",
    "effective_date": "2026-01-01",
    "terms": [
        "Text-to-speech output is generated using pre-trained speaker embeddings from public datasets.",
        "Users must NOT use synthesized voice to impersonate real individuals without consent.",
        "Generated audio must NOT be used for fraud, deception, or harassment.",
        "Users accept responsibility for downstream use of generated audio.",
        "Commercial use of synthesized speech requires appropriate licensing.",
    ],
    "prohibited_uses": [
        "Creating deepfake audio of real persons without consent",
        "Generating misleading or fraudulent audio content",
        "Bypassing voice-based authentication systems",
        "Creating non-consensual intimate or harmful content",
    ],
}


def get_consent_framework() -> dict:
    """Return the voice synthesis consent framework."""
    return CONSENT_TERMS


# ------------------------------------------------------------------ #
#  AI Model Transparency Card
# ------------------------------------------------------------------ #
def get_model_card() -> dict:
    """
    Return a comprehensive model transparency card
    documenting all AI models used in the system.
    """
    return {
        "system_name": "AI Speech Intelligence Platform v2.0",
        "models": [
            {
                "name": "OpenAI Whisper Medium",
                "model_id": "openai/whisper-medium",
                "task": "Automatic Speech Recognition (ASR)",
                "training_data": "680,000 hours of multilingual web audio",
                "languages": "99 languages supported",
                "known_limitations": [
                    "Lower accuracy on accented speech",
                    "May hallucinate on silence or background music",
                    "Performance degrades with heavy noise",
                ],
                "license": "MIT",
            },
            {
                "name": "Facebook BART Large CNN",
                "model_id": "facebook/bart-large-cnn",
                "task": "Abstractive Text Summarization",
                "training_data": "CNN/DailyMail news articles",
                "known_limitations": [
                    "Optimized for news-style text; may underperform on conversational speech",
                    "Max input length of 1024 tokens",
                ],
                "license": "MIT",
            },
            {
                "name": "DistilBERT SST-2",
                "model_id": "distilbert-base-uncased-finetuned-sst-2-english",
                "task": "Sentiment Analysis",
                "training_data": "Stanford Sentiment Treebank (SST-2)",
                "known_limitations": [
                    "Binary sentiment only (positive/negative)",
                    "English-only",
                    "May not capture nuanced or sarcastic sentiment",
                ],
                "license": "Apache 2.0",
            },
            {
                "name": "Microsoft SpeechT5 TTS",
                "model_id": "microsoft/speecht5_tts",
                "task": "Text-to-Speech Synthesis",
                "training_data": "LibriTTS dataset",
                "known_limitations": [
                    "English-only output",
                    "Requires speaker embeddings",
                    "May produce artifacts on long text",
                ],
                "license": "MIT",
            },
            {
                "name": "Helsinki-NLP OPUS-MT",
                "model_id": "Helsinki-NLP/opus-mt-en-*",
                "task": "Machine Translation",
                "training_data": "OPUS parallel corpus",
                "known_limitations": [
                    "Quality varies by language pair",
                    "May struggle with domain-specific terminology",
                ],
                "license": "CC-BY-4.0",
            },
        ],
        "ethical_considerations": [
            "All processing happens locally — no data sent to external servers",
            "Audio data is not stored after processing",
            "Users should verify transcription accuracy for critical applications",
            "Voice synthesis should only be used with appropriate consent",
        ],
    }


# ------------------------------------------------------------------ #
#  Usage Audit Logging
# ------------------------------------------------------------------ #
_audit_log = []


def log_usage_event(event_type: str, details: dict = None):
    """
    Log a usage event for audit purposes.

    Args:
        event_type: Type of event (e.g., 'transcription', 'tts', 'translation').
        details: Additional event details.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event_type": event_type,
        "details": details or {},
    }
    _audit_log.append(entry)
    logger.info(f"Audit: {event_type} — {details}")


def get_audit_log() -> list:
    """Return the current session's audit log."""
    return list(_audit_log)


def clear_audit_log():
    """Clear the audit log."""
    _audit_log.clear()
