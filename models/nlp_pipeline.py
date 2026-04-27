"""
NLP Processing Pipeline
Provides: summarization, action item extraction, sentiment analysis,
translation, and key topic extraction from transcribed text.
Uses Hugging Face transformers models.
"""

import re
import os
from contextlib import contextmanager
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM


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
        overall = sentiment(truncated)[0]

        # Per-sentence breakdown
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
        details = []
        max_sentences = 8 if self.quick_mode else 20
        for sent in sentences[:max_sentences]:
            result = sentiment(sent[:512])[0]
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
