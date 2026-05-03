import numpy as np

from src.models.whisper_model import WhisperASR


def test_merge_with_overlap_removes_duplicate_words():
    prev = "we need to finalize the sprint plan by friday"
    new = "the sprint plan by friday and assign owners"
    merged = WhisperASR._merge_with_overlap(prev, new)
    assert merged == "we need to finalize the sprint plan by friday and assign owners"


def test_merge_with_overlap_handles_empty_inputs():
    assert WhisperASR._merge_with_overlap("", "hello world") == "hello world"
    assert WhisperASR._merge_with_overlap("hello world", "") == "hello world"


def test_chunked_path_selected_for_long_audio(monkeypatch):
    model = WhisperASR(device="cpu")
    model.load_model = lambda: None

    called = {"chunked": False, "single": False}

    def fake_chunked(*args, **kwargs):
        called["chunked"] = True
        return "chunked transcript", 1.2

    def fake_single(*args, **kwargs):
        called["single"] = True
        return "single transcript", 0.3

    model._transcribe_chunked = fake_chunked
    model._transcribe_single_pass = fake_single

    # 31 seconds at 16kHz should go chunked path
    audio = np.zeros(31 * 16000, dtype=np.float32)
    result = model.transcribe(audio, sampling_rate=16000, language="en")

    assert called["chunked"] is True
    assert called["single"] is False
    assert result["text"] == "chunked transcript"
