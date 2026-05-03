"""
Utility script to pre-download models to local cache.
"""

import torch
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    SpeechT5Processor, 
    SpeechT5ForTextToSpeech, 
    SpeechT5HifiGan
)
from datasets import load_dataset
import os

def download_whisper_medium():
    model_id = "openai/whisper-medium"
    print(f"Downloading {model_id}...")
    WhisperProcessor.from_pretrained(model_id)
    WhisperForConditionalGeneration.from_pretrained(model_id)
    print(f"Finished downloading {model_id}")

def download_speecht5():
    tts_id = "microsoft/speecht5_tts"
    vocoder_id = "microsoft/speecht5_hifigan"
    dataset_id = "Matthijs/cmu-arctic-xvectors"
    
    print(f"Downloading {tts_id}...")
    SpeechT5Processor.from_pretrained(tts_id)
    SpeechT5ForTextToSpeech.from_pretrained(tts_id)
    
    print(f"Downloading {vocoder_id}...")
    SpeechT5HifiGan.from_pretrained(vocoder_id)
    
    print(f"Downloading speaker embeddings dataset {dataset_id}...")
    load_dataset(dataset_id, split="validation")
    
    print("Finished downloading SpeechT5 components")

if __name__ == "__main__":
    print("Starting model pre-download...")
    try:
        download_whisper_medium()
        download_speecht5()
        print("\nAll models downloaded successfully!")
    except Exception as e:
        print(f"\nError downloading models: {e}")
