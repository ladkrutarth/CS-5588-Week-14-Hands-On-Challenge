"""
NLP Processing Pipeline — Enhanced for Week 15
Provides: summarization, action item extraction, sentiment analysis,
translation, key topic extraction, meeting minutes generation,
and speaker turn segmentation from transcribed text.
Uses Hugging Face transformers models.
"""

import re
import os
import logging
from contextlib import contextmanager
from datetime import datetime
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class NLPPipeline:
    """
    Text analysis pipeline for processing Whisper transcriptions.
    All models are loaded on-demand to conserve GPU memory.
    """

    def __init__(self, device: str = "cpu", quick_mode: bool = True):
        """
        Args:
            device: Compute device for NLP models.
            quick_mode: Use faster, lighter processing for interactive UX.
        """
        self.device = device
        self.quick_mode = quick_mode
        self._summarizer = None
        self._sentiment = None
        self._translator = None
        self._seq2seq_models = {}
        self._seq2seq_tokenizers = {}

    @staticmethod
    @contextmanager
    def _without_proxy_env():
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

    def _safe_pipeline(self, task: str, model_id: str):
        device_id = 0 if self.device == "cuda" else -1
        try:
            with self._without_proxy_env():
                return pipeline(task, model=model_id, device=device_id)
        except Exception:
            return pipeline(task, model=model_id, device=device_id, local_files_only=True)

    # ------------------------------------------------------------------ #
    #  Summarization
    # ------------------------------------------------------------------ #
    def _get_summarizer(self):
        model_id = "facebook/bart-large-cnn"
        return self._get_seq2seq_pair(model_id)

    def _get_seq2seq_pair(self, model_id: str):
        if model_id not in self._seq2seq_tokenizers:
            try:
                with self._without_proxy_env():
                    self._seq2seq_tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
            except Exception:
                self._seq2seq_tokenizers[model_id] = AutoTokenizer.from_pretrained(
                    model_id,
                    local_files_only=True,
                )
        if model_id not in self._seq2seq_models:
            try:
                with self._without_proxy_env():
                    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            except Exception:
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, local_files_only=True)
            if self.device in {"cuda", "mps"}:
                model = model.to(self.device)
            model.eval()
            self._seq2seq_models[model_id] = model
        return self._seq2seq_tokenizers[model_id], self._seq2seq_models[model_id]

    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        """
        Generate an abstractive summary of the input text.

        Args:
            text: Input text to summarize.
            max_length: Maximum summary length in tokens.
            min_length: Minimum summary length in tokens.

        Returns:
            Summary string.
        """
        if not text or len(text.split()) < 30:
            return text  # Too short to summarize

        if self.quick_mode:
            return self._fast_extractive_summary(text, max_sentences=3)

        tokenizer, model = self._get_summarizer()

        # BART-large-CNN has a 1024 token limit; chunk if needed
        chunks = self._chunk_text(text, max_words=800)
        summaries = []

        for chunk in chunks:
            generated = self._run_seq2seq(
                tokenizer,
                model,
                chunk,
                max_length=max_length,
                min_length=min(min_length, len(chunk.split()) // 2),
            )
            summaries.append(generated.strip())

        joined = " ".join([s for s in summaries if s])
        return joined if joined else text

    # ------------------------------------------------------------------ #
    #  Action Items Extraction
    # ------------------------------------------------------------------ #
    def extract_action_items(self, text: str) -> list:
        """
        Extract actionable items from text using rule-based + keyword matching.

        Args:
            text: Input text.

        Returns:
            List of action item strings.
        """
        action_patterns = [
            r"(?:we |I |you |they |he |she )?(?:need to|should|must|have to|will|going to|plan to|want to|let's|let us)\s+(.+?)(?:\.|$)",
            r"(?:action item|todo|to-do|task|follow up|follow-up)[:\s]+(.+?)(?:\.|$)",
            r"(?:please|kindly)\s+(.+?)(?:\.|$)",
            r"(?:make sure|ensure|remember to|don't forget to)\s+(.+?)(?:\.|$)",
            r"(?:deadline|due date|by)\s+(.+?)(?:\.|$)",
            r"(?:assign|assigned to|responsible for)\s+(.+?)(?:\.|$)",
        ]

        sentences = re.split(r'[.!?]+', text)
        action_items = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            for pattern in action_patterns:
                matches = re.findall(pattern, sentence, re.IGNORECASE)
                if matches:
                    # Use the full sentence as the action item
                    item = sentence.strip()
                    if item and item not in action_items and len(item) > 10:
                        action_items.append(item)
                    break

        # Deduplicate
        seen = set()
        unique = []
        for item in action_items:
            key = item.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(item)

        return unique if unique else ["No specific action items detected."]

    # ------------------------------------------------------------------ #
    #  Sentiment Analysis
    # ------------------------------------------------------------------ #
    def _get_sentiment(self):
        if self._sentiment is None:
            self._sentiment = self._safe_pipeline(
                "sentiment-analysis",
                "distilbert-base-uncased-finetuned-sst-2-english",
            )
        return self._sentiment

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyze overall sentiment and per-sentence breakdown.

        Args:
            text: Input text.

        Returns:
            dict with 'overall' sentiment and 'details' per sentence.
        """
        sentiment = self._get_sentiment()

        # Overall sentiment (truncate to model max)
        word_cap = 180 if self.quick_mode else 500
        truncated = " ".join(text.split()[:word_cap])
        overall = sentiment(
            truncated,
            truncation=True,
            max_length=512,
        )[0]

        # Per-sentence breakdown
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
        details = []
        max_sentences = 8 if self.quick_mode else 20
        for sent in sentences[:max_sentences]:
            result = sentiment(
                sent,
                truncation=True,
                max_length=512,
            )[0]
            details.append({
                "sentence": sent,
                "label": result["label"],
                "score": round(result["score"], 3),
            })

        # Compute aggregate stats
        pos_count = sum(1 for d in details if d["label"] == "POSITIVE")
        neg_count = sum(1 for d in details if d["label"] == "NEGATIVE")
        total = len(details) if details else 1

        return {
            "overall": {
                "label": overall["label"],
                "score": round(overall["score"], 3),
            },
            "breakdown": {
                "positive_pct": round(pos_count / total * 100, 1),
                "negative_pct": round(neg_count / total * 100, 1),
                "total_sentences": total,
            },
            "details": details,
        }

    # ------------------------------------------------------------------ #
    #  Translation
    # ------------------------------------------------------------------ #
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

    def translate(self, text: str, target_language: str = "Spanish") -> str:
        """
        Translate English text to a target language.

        Args:
            text: English text to translate.
            target_language: Target language name.

        Returns:
            Translated text string.
        """
        model_id = self.TRANSLATION_MODELS.get(target_language)
        if not model_id:
            return f"[Translation to {target_language} is not supported]"

        if self._translator is None:
            self._translator = {}
        try:
            if model_id not in self._translator:
                self._translator[model_id] = self._get_seq2seq_pair(model_id)
            translator = self._translator[model_id]
        except Exception:
            return (
                f"[Translation unavailable for {target_language}. "
                "Model is not cached locally and Hugging Face access is blocked.]"
            )

        # In quick mode, translate a smaller slice for faster turnaround.
        if self.quick_mode:
            text = " ".join(text.split()[:140])

        # Chunk text for translation
        chunks = self._chunk_text(text, max_words=300 if self.quick_mode else 400)
        translated = []

        for chunk in chunks:
            tokenizer, model = translator
            try:
                generated = self._run_seq2seq(
                    tokenizer,
                    model,
                    chunk,
                    max_length=512,
                    min_length=10,
                )
                translated.append(generated.strip())
            except Exception:
                return (
                    f"[Translation generation failed for {target_language}. "
                    "Disable translation in offline mode or retry when model download works.]"
                )

        joined = " ".join([t for t in translated if t])
        return joined if joined else "[Translation did not return text.]"

    # ------------------------------------------------------------------ #
    #  Key Topics Extraction
    # ------------------------------------------------------------------ #
    def extract_key_topics(self, text: str, top_n: int = 10) -> list:
        """
        Extract key topics using TF-based keyword extraction.

        Args:
            text: Input text.
            top_n: Number of top keywords to return.

        Returns:
            List of (keyword, frequency) tuples.
        """
        # Simple TF-based extraction (no extra dependencies needed)
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "from", "is", "it", "that", "this",
            "was", "are", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may",
            "might", "shall", "can", "need", "dare", "ought", "used", "i",
            "me", "my", "we", "our", "you", "your", "he", "him", "his",
            "she", "her", "they", "them", "their", "what", "which", "who",
            "whom", "whose", "when", "where", "why", "how", "not", "no",
            "nor", "as", "if", "then", "than", "too", "very", "so", "just",
            "about", "above", "after", "again", "all", "also", "am", "any",
            "because", "before", "between", "both", "each", "few", "get",
            "got", "here", "into", "its", "like", "more", "most", "much",
            "must", "now", "only", "other", "out", "over", "own", "same",
            "some", "still", "such", "take", "tell", "there", "these",
            "thing", "think", "those", "through", "up", "us", "want", "well",
            "going", "yeah", "okay", "right", "know", "really", "actually",
            "uh", "um", "kind", "lot", "say", "said", "one", "two", "make",
        }

        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        freq = {}
        for w in words:
            if w not in stop_words:
                freq[w] = freq.get(w, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        effective_top_n = 7 if self.quick_mode else top_n
        return sorted_words[:effective_top_n]

    # ------------------------------------------------------------------ #
    #  Speaker Turn Segmentation (Simulated Diarization)
    # ------------------------------------------------------------------ #
    def segment_speaker_turns(self, text: str) -> list:
        """
        Simulate speaker diarization by detecting speaker turn boundaries
        based on discourse markers, sentence structure, and topic shifts.

        This is a rule-based approximation — real diarization requires
        audio-level analysis, but this provides useful structure for
        meeting transcripts.

        Args:
            text: Transcribed text.

        Returns:
            List of dicts with speaker label, text segment, and word count.
        """
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return [{"speaker": "Speaker 1", "text": text, "word_count": len(text.split())}]

        # Detect potential speaker changes based on discourse markers
        turn_markers = [
            r"^(so|well|okay|alright|right|now|actually|basically|honestly|look|listen)\b",
            r"^(I think|I believe|In my opinion|From my perspective)",
            r"^(yes|no|yeah|nah|absolutely|definitely|sure|exactly)\b",
            r"^(but|however|although|on the other hand|conversely)",
            r"^(thank you|thanks|great|perfect|wonderful|excellent)",
        ]

        segments = []
        current_speaker = 1
        current_text = []
        sentences_since_change = 0
        max_speakers = 4  # Limit simulated speakers

        for sent in sentences:
            is_turn = False

            # Check for turn markers
            for pattern in turn_markers:
                if re.search(pattern, sent, re.IGNORECASE):
                    if sentences_since_change >= 2:  # Minimum 2 sentences per turn
                        is_turn = True
                    break

            # Also trigger on significant pause indicators
            if sent.startswith("...") or sent.startswith("-"):
                if sentences_since_change >= 2:
                    is_turn = True

            if is_turn and current_text:
                segment_text = " ".join(current_text)
                segments.append({
                    "speaker": f"Speaker {current_speaker}",
                    "text": segment_text,
                    "word_count": len(segment_text.split()),
                })
                current_speaker = (current_speaker % max_speakers) + 1
                current_text = [sent]
                sentences_since_change = 0
            else:
                current_text.append(sent)
                sentences_since_change += 1

        # Add final segment
        if current_text:
            segment_text = " ".join(current_text)
            segments.append({
                "speaker": f"Speaker {current_speaker}",
                "text": segment_text,
                "word_count": len(segment_text.split()),
            })

        return segments

    # ------------------------------------------------------------------ #
    #  Meeting Minutes Generator
    # ------------------------------------------------------------------ #
    def generate_meeting_minutes(
        self,
        transcript: str,
        summary: str = None,
        action_items: list = None,
        sentiment: dict = None,
        topics: list = None,
        metadata: dict = None,
    ) -> str:
        """
        Generate structured meeting minutes document.

        Args:
            transcript: Full transcription.
            summary: Pre-computed summary (optional).
            action_items: Pre-computed action items (optional).
            sentiment: Pre-computed sentiment (optional).
            topics: Pre-computed topics (optional).
            metadata: Additional meeting metadata (optional).

        Returns:
            Formatted meeting minutes as markdown string.
        """
        now = datetime.now()
        meta = metadata or {}

        minutes = []
        minutes.append("# 📋 Meeting Minutes")
        minutes.append(f"**Date:** {now.strftime('%B %d, %Y')}")
        minutes.append(f"**Time:** {now.strftime('%I:%M %p')}")
        if meta.get("duration"):
            minutes.append(f"**Duration:** {meta['duration']}")
        if meta.get("language"):
            minutes.append(f"**Language:** {meta['language']}")
        minutes.append(f"**Generated by:** AI Speech Intelligence Platform v2.0")
        minutes.append("")

        # Summary
        minutes.append("## 📝 Executive Summary")
        if summary and summary != transcript:
            minutes.append(summary)
        else:
            # Auto-generate a quick summary
            summary = self._fast_extractive_summary(transcript, max_sentences=3)
            minutes.append(summary)
        minutes.append("")

        # Key Topics
        if topics:
            minutes.append("## 🏷️ Key Topics Discussed")
            for word, count in topics[:7]:
                minutes.append(f"- **{word.title()}** (mentioned {count}x)")
            minutes.append("")

        # Speaker Segments
        segments = self.segment_speaker_turns(transcript)
        if len(segments) > 1:
            minutes.append("## 👥 Discussion Summary")
            for seg in segments:
                preview = seg["text"][:200] + "..." if len(seg["text"]) > 200 else seg["text"]
                minutes.append(f"**{seg['speaker']}** ({seg['word_count']} words):")
                minutes.append(f"> {preview}")
                minutes.append("")

        # Action Items
        minutes.append("## ✅ Action Items")
        if action_items:
            for i, item in enumerate(action_items, 1):
                if item != "No specific action items detected.":
                    minutes.append(f"{i}. {item}")
        else:
            minutes.append("- No specific action items detected.")
        minutes.append("")

        # Sentiment
        if sentiment:
            minutes.append("## 💭 Meeting Tone")
            overall = sentiment.get("overall", {})
            label = overall.get("label", "N/A")
            score = overall.get("score", 0)
            bd = sentiment.get("breakdown", {})
            minutes.append(f"- **Overall Sentiment:** {label} ({score:.0%} confidence)")
            minutes.append(f"- **Positive statements:** {bd.get('positive_pct', 0):.0f}%")
            minutes.append(f"- **Negative statements:** {bd.get('negative_pct', 0):.0f}%")
            minutes.append("")

        # Full Transcript
        minutes.append("## 📄 Full Transcript")
        minutes.append(f"> {transcript}")
        minutes.append("")

        minutes.append("---")
        minutes.append(f"*Auto-generated on {now.strftime('%Y-%m-%d %H:%M:%S')} by AI Speech Intelligence Platform*")

        return "\n".join(minutes)

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #
    @staticmethod
    def _chunk_text(text: str, max_words: int = 800) -> list:
        """Split text into chunks of approximately max_words."""
        words = text.split()
        if len(words) <= max_words:
            return [text]

        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i : i + max_words])
            chunks.append(chunk)
        return chunks

    def _run_seq2seq(self, tokenizer, model, text: str, max_length: int, min_length: int) -> str:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        if self.device in {"cuda", "mps"}:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_length,
                min_length=max(1, min_length),
                do_sample=False,
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True)

    @staticmethod
    def _fast_extractive_summary(text: str, max_sentences: int = 3) -> str:
        """
        Very fast extractive summary for interactive mode.
        """
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return text
        return " ".join(sentences[:max_sentences])
