"""
Centralized Configuration — AI Speech Intelligence Platform
All model IDs, hyperparameters, feature flags, and system constants.
"""

import os

# ------------------------------------------------------------------ #
#  Model Configuration
# ------------------------------------------------------------------ #
WHISPER_MODEL_ID = "openai/whisper-medium"
SUMMARIZATION_MODEL_ID = "facebook/bart-large-cnn"
SENTIMENT_MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
TTS_MODEL_ID = "microsoft/speecht5_tts"
TTS_VOCODER_ID = "microsoft/speecht5_hifigan"
TTS_SPEAKER_DATASET = "Matthijs/cmu-arctic-xvectors"

# Translation model mapping
TRANSLATION_MODELS = {
    "Spanish": "Helsinki-NLP/opus-mt-en-es",
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Chinese": "Helsinki-NLP/opus-mt-en-zh",
    "Japanese": "Helsinki-NLP/opus-mt-en-jap",
    "Arabic": "Helsinki-NLP/opus-mt-en-ar",
    "Hindi": "Helsinki-NLP/opus-mt-en-hi",
    "Portuguese": "Helsinki-NLP/opus-mt-en-ROMANCE",
    "Russian": "Helsinki-NLP/opus-mt-en-ru",
}

# ------------------------------------------------------------------ #
#  Audio Configuration
# ------------------------------------------------------------------ #
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_DURATION_SECONDS = 600  # 10 minutes
SILENCE_TRIM_DB = 30
NORMALIZATION_TARGET_DB = -20.0

# ------------------------------------------------------------------ #
#  ASR Configuration
# ------------------------------------------------------------------ #
DEFAULT_BEAM_SIZE = 1
IMPROVED_BEAM_SIZE = 5
MAX_NEW_TOKENS = 220
DEFAULT_TEMPERATURE = 0.0

# ------------------------------------------------------------------ #
#  NLP Configuration
# ------------------------------------------------------------------ #
SUMMARY_MAX_LENGTH = 150
SUMMARY_MIN_LENGTH = 40
SUMMARY_MAX_WORDS_CHUNK = 800
SENTIMENT_WORD_CAP_QUICK = 180
SENTIMENT_WORD_CAP_FULL = 500
SENTIMENT_MAX_SENTENCES_QUICK = 8
SENTIMENT_MAX_SENTENCES_FULL = 20
TOPIC_TOP_N_QUICK = 7
TOPIC_TOP_N_FULL = 10

# ------------------------------------------------------------------ #
#  TTS Configuration
# ------------------------------------------------------------------ #
TTS_MAX_CHARS = 500
TTS_DEFAULT_SPEAKER_ID = 7306
TTS_SAMPLE_RATE = 16000

# ------------------------------------------------------------------ #
#  Evaluation Configuration
# ------------------------------------------------------------------ #
EVAL_NUM_RUNS = 3  # Number of runs for latency averaging

# ------------------------------------------------------------------ #
#  Business / Pricing Tiers
# ------------------------------------------------------------------ #
PRICING_TIERS = {
    "Free": {
        "price": "$0/mo",
        "features": [
            "5 transcriptions/day",
            "Max 2 min audio",
            "Basic summary",
            "Community support",
        ],
        "color": "#667eea",
    },
    "Pro": {
        "price": "$29/mo",
        "features": [
            "Unlimited transcriptions",
            "Max 60 min audio",
            "Full NLP pipeline",
            "TTS voice output",
            "Priority support",
            "Export to PDF/JSON",
        ],
        "color": "#764ba2",
        "popular": True,
    },
    "Enterprise": {
        "price": "Custom",
        "features": [
            "Everything in Pro",
            "On-premise deployment",
            "Custom model fine-tuning",
            "SLA guarantee",
            "Dedicated support",
            "API access",
            "Audit logging",
        ],
        "color": "#f093fb",
    },
}

# ------------------------------------------------------------------ #
#  Responsible AI
# ------------------------------------------------------------------ #
SENSITIVE_PATTERNS = [
    r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b",  # SSN
    r"\b\d{16}\b",  # Credit card
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
    r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",  # Phone number
]

BIAS_WARNING_THRESHOLDS = {
    "short_transcript_words": 5,  # Warn if transcript is suspiciously short
    "repetition_ratio": 0.5,  # Warn if >50% of words repeat
}

# ------------------------------------------------------------------ #
#  UI Theme
# ------------------------------------------------------------------ #
THEME = {
    "primary_gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "secondary_gradient": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
    "accent_gradient": "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    "dark_bg": "#0a0a1a",
    "card_bg": "rgba(26, 26, 46, 0.8)",
    "card_border": "rgba(102, 126, 234, 0.2)",
    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "success": "#4ade80",
    "warning": "#fbbf24",
    "error": "#f87171",
}

# ------------------------------------------------------------------ #
#  Logging
# ------------------------------------------------------------------ #
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ------------------------------------------------------------------ #
#  Application Metadata
# ------------------------------------------------------------------ #
APP_NAME = "AI Speech Intelligence Platform"
APP_VERSION = "2.0.0"
APP_DESCRIPTION = "Real-Time Multilingual Meeting Intelligence Platform"
APP_ICON = "🎙️"
