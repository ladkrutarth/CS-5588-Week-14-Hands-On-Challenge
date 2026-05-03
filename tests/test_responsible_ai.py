from src.utils.responsible_ai import detect_transcription_bias


def test_hindi_common_words_do_not_trigger_repetition_false_positive():
    transcript = "यह एक बैठक है और यह परियोजना के लिए महत्वपूर्ण है और टीम के साथ प्रगति पर चर्चा है"
    result = detect_transcription_bias(transcript, audio_duration_seconds=35, language="hi")
    types = {w["type"] for w in result["warnings"]}
    assert "high_repetition" not in types


def test_actual_hallucination_like_repetition_is_detected():
    transcript = "subscribe subscribe subscribe subscribe subscribe subscribe thank you for watching"
    result = detect_transcription_bias(transcript, audio_duration_seconds=30, language="en")
    types = {w["type"] for w in result["warnings"]}
    assert "high_repetition" in types or "possible_hallucination" in types
