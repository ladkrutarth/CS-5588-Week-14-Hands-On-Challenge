"""
TTS Engine - Text-to-Speech using Microsoft SpeechT5
Uses microsoft/speecht5_tts and microsoft/speecht5_hifigan.
Requires speaker embeddings for synthesis.
"""

import torch
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import os
import time

class SpeechTTS:
    """
    Text-to-Speech engine using Microsoft SpeechT5.
    """
    TTS_MODEL_ID = "microsoft/speecht5_tts"
    VOCODER_ID = "microsoft/speecht5_hifigan"
    SPEAKER_EMBEDDINGS_DATASET = "Matthijs/cmu-arctic-xvectors"

    def __init__(self, device: str = None):
        self.device = device or self._detect_device()
        self.processor = None
        self.model = None
        self.vocoder = None
        self.embeddings_dataset = None
        self._loaded = False
        print(f"[SpeechTTS] Target device: {self.device}")

    @staticmethod
    def _detect_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self):
        if self._loaded:
            return

        print(f"[SpeechTTS] Loading models on {self.device}...")
        start = time.time()

        self.processor = SpeechT5Processor.from_pretrained(self.TTS_MODEL_ID)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(self.TTS_MODEL_ID)
        self.vocoder = SpeechT5HifiGan.from_pretrained(self.VOCODER_ID)
        
        # Move to device
        self.model.to(self.device)
        self.vocoder.to(self.device)
        
        # Load speaker embeddings (defaulting to the first one in the dataset)
        print("[SpeechTTS] Loading speaker embeddings...")
        self.embeddings_dataset = load_dataset(self.SPEAKER_EMBEDDINGS_DATASET, split="validation")
        
        self.model.eval()
        self._loaded = True
        elapsed = time.time() - start
        print(f"[SpeechTTS] Models loaded in {elapsed:.2f}s")

    def synthesize(self, text: str, speaker_id: int = 7306) -> np.ndarray:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize.
            speaker_id: ID of the speaker embedding to use.
            
        Returns:
            numpy array containing the audio waveform.
        """
        self.load_model()
        
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get speaker embeddings
        # speaker_id 7306 is a common default from the dataset
        speaker_embeddings = torch.tensor(self.embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            speech = self.model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder)
            
        return speech.cpu().numpy()

    def get_info(self) -> dict:
        return {
            "device": self.device,
            "tts_model": self.TTS_MODEL_ID,
            "vocoder": self.VOCODER_ID,
            "loaded": self._loaded
        }
