from src.utils.tts_engine import LocalTTSEngine
from src.utils.tts_engine import VoiceOption


def test_sanitize_text_collapses_whitespace():
    text = "Hello   world\n\nthis\tis   test"
    assert LocalTTSEngine.sanitize_text(text) == "Hello world this is test"


def test_sanitize_text_handles_empty():
    assert LocalTTSEngine.sanitize_text("") == ""
    assert LocalTTSEngine.sanitize_text("   ") == ""


def test_professional_voice_filter_prefers_language():
    engine = LocalTTSEngine()
    engine.list_voices = lambda: [
        VoiceOption(id="Lekha", name="Lekha", gender="female", locale="hi_IN", language="hi"),
        VoiceOption(id="Samantha", name="Samantha", gender="female", locale="en_US", language="en"),
    ]
    voices = engine.get_professional_voice_options(gender="female", language="hi")
    assert len(voices) == 1
    assert voices[0].id == "Lekha"


def test_language_default_voice_contains_hindi():
    assert LocalTTSEngine.LANGUAGE_DEFAULT_VOICE["hi"] == "Lekha"
