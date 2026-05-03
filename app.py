"""
AI Speech Intelligence Platform v2.0
Real-Time Multilingual Meeting Intelligence Platform
Enhanced for Week 15 Final Challenge
"""

import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import os
import json

# Import local modules
from src import config
from src.models.whisper_model import WhisperASR
from src.models.nlp_pipeline import NLPPipeline
from src.models.evaluation import TranscriptionEvaluator
from src.models.tts_model import SpeechTTS
from src.utils.audio_utils import load_audio, get_audio_info, preprocess_audio, compute_waveform_data
from src.utils.noise_augmentation import apply_noise_preset, NOISE_PRESETS
from src.utils.responsible_ai import scan_sensitive_content, detect_transcription_bias, get_model_card, log_usage_event
from src.utils.tts_engine import LocalTTSEngine

# -- Page Config --
st.set_page_config(
    page_title=f"{config.APP_NAME} v{config.APP_VERSION}",
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Premium CSS --
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
* {{ font-family: 'Outfit', sans-serif; }}

.stApp {{
    background-color: {config.THEME['dark_bg']};
    color: {config.THEME['text_primary']};
}}

.main-header {{
    background: {config.THEME['primary_gradient']};
    padding: 3rem 2rem;
    border-radius: 24px;
    margin-bottom: 2.5rem;
    color: white;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}}

.glass-card {{
    background: {config.THEME['card_bg']};
    backdrop-filter: blur(10px);
    border: 1px solid {config.THEME['card_border']};
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: transform 0.3s ease;
}}

.glass-card:hover {{
    transform: translateY(-5px);
    border-color: rgba(102, 126, 234, 0.5);
}}

.metric-value {{
    font-size: 2rem;
    font-weight: 700;
    background: {config.THEME['accent_gradient']};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}}

.status-badge {{
    padding: 0.2rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    display: inline-block;
    margin-bottom: 0.5rem;
}}

.status-success {{ background: rgba(74, 222, 128, 0.2); color: #4ade80; border: 1px solid #4ade80; }}
.status-warning {{ background: rgba(251, 191, 36, 0.2); color: #fbbf24; border: 1px solid #fbbf24; }}

.action-item {{
    border-left: 4px solid #667eea;
    padding: 1rem;
    margin: 0.8rem 0;
    background: rgba(255,255,255,0.03);
    border-radius: 0 12px 12px 0;
}}

/* Custom Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 12px; }}
.stTabs [data-baseweb="tab"] {{
    background: rgba(255,255,255,0.05);
    border-radius: 12px 12px 0 0;
    padding: 10px 24px;
    color: {config.THEME['text_secondary']};
}}
.stTabs [aria-selected="true"] {{
    background: {config.THEME['primary_gradient']} !important;
    color: white !important;
}}
</style>
""", unsafe_allow_html=True)

# -- Resource Caching --
@st.cache_resource
def get_whisper():
    model = WhisperASR()
    model.load_model()
    return model

@st.cache_resource
def get_nlp(device, quick_mode=True):
    return NLPPipeline(device=device, quick_mode=quick_mode)

@st.cache_resource
def get_tts(device):
    return SpeechTTS(device=device)

@st.cache_resource
def get_local_tts():
    return LocalTTSEngine()

def get_evaluator():
    return TranscriptionEvaluator()

# -- Session State --
if "history" not in st.session_state:
    st.session_state.history = []
if "current_result" not in st.session_state:
    st.session_state.current_result = None

# -- Sidebar Settings --
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microphone.png", width=80)
    st.title("Settings")
    
    with st.expander("🌍 Language & Task", expanded=True):
        whisper_language_codes = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Chinese": "zh",
            "Japanese": "ja",
            "Arabic": "ar",
            "Hindi": "hi",
            "Portuguese": "pt",
            "Russian": "ru",
        }
        src_lang = st.selectbox("Source Language", ["Auto Detect"] + list(whisper_language_codes.keys()))
        task = st.radio("Task", ["Transcribe", "Translate to English"])
        lang_code = whisper_language_codes.get(src_lang) if src_lang != "Auto Detect" else None
    
    with st.expander("⚡ Performance", expanded=False):
        beam_size = st.slider("Beam Size", 1, 10, config.DEFAULT_BEAM_SIZE)
        use_fp16 = st.checkbox("Use FP16 (GPU only)", value=True)
        quick_nlp = st.checkbox("Quick NLP Mode", value=True)
    
    with st.expander("🛡️ Safety & Ethics", expanded=False):
        enable_pii_scan = st.checkbox("Scan for Sensitive Data", value=True)
        show_model_card = st.button("View AI Transparency Card")
    
    st.markdown("---")
    st.info(f"System: **{config.APP_NAME}**\nVersion: **{config.APP_VERSION}**")

# -- Header --
st.markdown(f"""
<div class="main-header">
    <h1>{config.APP_ICON} {config.APP_NAME}</h1>
    <p>{config.APP_DESCRIPTION}</p>
</div>
""", unsafe_allow_html=True)

# -- Mode Switch (tab-like) --
app_mode = st.radio(
    "Workspace",
    ["🎙️ Speech Intelligence", "🗣️ Text-to-Speech"],
    horizontal=True,
)

if app_mode == "🗣️ Text-to-Speech":
    st.markdown("### 🗣️ Text-to-Speech (Local)")
    st.caption("Dedicated multilingual TTS workspace with professional male/female voice options.")
    tts_language_codes = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Portuguese": "pt",
        "Hindi": "hi",
        "Japanese": "ja",
        "Chinese": "zh",
        "Arabic": "ar",
        "Russian": "ru",
    }

    tts_text = st.text_area(
        "Enter text to synthesize",
        value="Welcome to the AI Speech Intelligence Platform.",
        height=120,
    )
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    with preset_col1:
        voice_gender = st.selectbox("Voice Gender", ["Female", "Male"], index=0)
    with preset_col2:
        tts_language = st.selectbox(
            "Speech Language",
            [
                "English", "Spanish", "French", "German", "Portuguese", "Hindi",
                "Japanese", "Chinese", "Arabic", "Russian"
            ],
            index=0,
        )
    with preset_col3:
        tts_rate = st.slider("Speech rate", min_value=120, max_value=220, value=165, step=5)
    selected_voice_id = None

    try:
        local_tts = get_local_tts()
        voices = local_tts.get_professional_voice_options(
            gender=voice_gender.lower(),
            language=tts_language_codes.get(tts_language, "en"),
        )
        if voices:
            labels = []
            for v in voices:
                parts = [v.name]
                if v.gender != "unknown":
                    parts.append(v.gender)
                if v.locale != "unknown":
                    parts.append(v.locale)
                labels.append(" | ".join(parts))
            selected_voice_name = st.selectbox("Professional Voice", labels, key="tts_voice_select")
            idx = labels.index(selected_voice_name)
            selected_voice_id = voices[idx].id
            st.caption("Tip: set language first, then choose a matching male/female voice.")
        else:
            st.warning("No matching voices found for this language/gender. Try English or switch gender.")
    except Exception as e:
        st.warning(f"Local TTS setup issue: {e}")

    if st.button("🔊 Generate TTS Audio", use_container_width=True):
        try:
            audio_bytes = get_local_tts().synthesize_to_wav_bytes(
                tts_text,
                rate=tts_rate,
                voice_id=selected_voice_id,
                language=tts_language_codes.get(tts_language, "en"),
            )
            st.session_state["tts_audio_bytes"] = audio_bytes
            st.success("Audio generated.")
        except Exception as e:
            st.error(f"TTS failed: {e}")

    if "tts_audio_bytes" in st.session_state:
        st.audio(st.session_state["tts_audio_bytes"], format="audio/wav")
        st.download_button(
            "⬇️ Download TTS WAV",
            data=st.session_state["tts_audio_bytes"],
            file_name="tts_output.wav",
            mime="audio/wav",
        )
    st.stop()

if show_model_card:
    st.info("### 🤖 AI Transparency Card")
    card = get_model_card()
    for m in card["models"]:
        with st.expander(f"Model: {m['name']}"):
            st.write(m)
    st.markdown("---")

# -- Main Interface --
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📥 Input Selection")
    input_type = st.tabs(["📁 File Upload", "🎤 Live Record"])
    
    audio_input = None
    with input_type[0]:
        uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a", "flac"])
        if uploaded_file:
            audio_input = uploaded_file.read()
            
    with input_type[1]:
        if hasattr(st, "audio_input"):
            mic_input = st.audio_input("Record now")
            if mic_input:
                audio_input = mic_input.read()
        else:
            st.warning("Microphone input not supported in this version.")

with col2:
    st.markdown("### 🛠️ Preprocessing")
    do_normalize = st.checkbox("Normalize Volume", value=True)
    do_bandpass = st.checkbox("Speech Enhancement Filter", value=True)
    do_trim = st.checkbox("Trim Silence", value=True)
    noise_test = st.selectbox("Inject Noise (Evaluation only)", list(NOISE_PRESETS.keys()))

# -- Execution --
if audio_input:
    st.markdown("---")
    
    if st.button("🚀 Process Intelligence Pipeline", type="primary", use_container_width=True):
        progress_bar = st.progress(0, "Loading Engine...")
        
        try:
            # 1. Load Audio
            progress_bar.progress(10, "Loading Audio...")
            raw_audio, sr = load_audio(audio_input)
            
            # 2. Preprocess
            progress_bar.progress(20, "Applying Signal Processing...")
            processed_audio = preprocess_audio(
                raw_audio, sr, trim=do_trim, normalize=do_normalize, bandpass=do_bandpass
            )
            if noise_test != "Clean (No Noise)":
                processed_audio = apply_noise_preset(processed_audio, noise_test)
                
            audio_info = get_audio_info(processed_audio, sr)
            waveform = compute_waveform_data(processed_audio, sr)
            
            # 3. Transcription
            progress_bar.progress(40, f"Transcribing with Whisper ({config.WHISPER_MODEL_ID})...")
            whisper = get_whisper()
            asr_start = time.time()
            asr_result = whisper.transcribe_with_settings(
                processed_audio, sr, language=lang_code, 
                beam_size=beam_size, task=task.lower()
            )
            asr_latency = time.time() - asr_start
            transcript = asr_result["text"]
            
            # 4. NLP Analysis
            progress_bar.progress(70, "Extracting Intelligence...")
            nlp = get_nlp(whisper.device, quick_mode=quick_nlp)
            
            summary = nlp.summarize(transcript)
            action_items = nlp.extract_action_items(transcript)
            sentiment = nlp.analyze_sentiment(transcript)
            topics = nlp.extract_key_topics(transcript)
            segments = nlp.segment_speaker_turns(transcript)
            
            # 5. Security Scan
            progress_bar.progress(90, "Security & Bias Scan...")
            security = scan_sensitive_content(transcript) if enable_pii_scan else {"has_sensitive_content": False}
            bias = detect_transcription_bias(
                transcript,
                audio_info["duration_seconds"],
                language=lang_code,
            )
            
            # Finalize
            progress_bar.progress(100, "Complete!")
            
            st.session_state.current_result = {
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "sentiment": sentiment,
                "topics": topics,
                "segments": segments,
                "audio_info": audio_info,
                "waveform": waveform,
                "security": security,
                "bias": bias,
                "latency": asr_latency,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Log event
            log_usage_event("intelligence_pipeline", {"duration": audio_info["duration_seconds"], "latency": asr_latency})
            
            st.balloons()
            
        except Exception as e:
            st.error(f"Pipeline Error: {str(e)}")
            st.exception(e)

# -- Results Dashboard --
if st.session_state.current_result:
    res = st.session_state.current_result
    
    # Waveform Header
    fig_wave = px.line(x=res["waveform"]["times"], y=res["waveform"]["amplitudes"],
                      labels={"x": "Time (s)", "y": "Amplitude"},
                      title="Audio Signal Analysis")
    fig_wave.update_layout(template="plotly_dark", height=200, margin=dict(l=0, r=0, t=30, b=0),
                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_wave, use_container_width=True)

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="glass-card"><h3>Duration</h3><p class="metric-value">{res["audio_info"]["duration_seconds"]}s</p></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="glass-card"><h3>Latency</h3><p class="metric-value">{res["latency"]:.2f}s</p></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="glass-card"><h3>SNR</h3><p class="metric-value">{res["audio_info"]["snr_estimate_db"]}dB</p></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="glass-card"><h3>Sentiment</h3><p class="metric-value">{res["sentiment"]["overall"]["label"]}</p></div>', unsafe_allow_html=True)

    # Tabs for detailed analysis
    tab_trans, tab_sum, tab_action, tab_speaker, tab_sec, tab_biz = st.tabs([
        "📝 Transcript", "📋 Summary", "✅ Actions", "👥 Speakers", "🛡️ Security", "💰 Business"
    ])
    
    with tab_trans:
        st.markdown(f'<div style="background:rgba(255,255,255,0.05); padding:20px; border-radius:15px; border:1px solid rgba(255,255,255,0.1);">{res["transcript"]}</div>', unsafe_allow_html=True)
        if res["bias"]["has_issues"]:
            st.warning("#### ⚠️ Quality Alerts")
            for w in res["bias"]["warnings"]:
                st.write(f"- **{w['type'].title()}**: {w['message']}")

    with tab_sum:
        st.markdown("### AI Summary")
        st.write(res["summary"])
        
        st.markdown("### 🏷️ Topics")
        topic_df = pd.DataFrame(res["topics"], columns=["Keyword", "Frequency"])
        fig_topics = px.bar(topic_df, x="Keyword", y="Frequency", color="Frequency", color_continuous_scale="Viridis")
        fig_topics.update_layout(template="plotly_dark", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_topics, use_container_width=True)

    with tab_action:
        st.markdown("### Action Items")
        for item in res["action_items"]:
            st.markdown(f'<div class="action-item">📌 {item}</div>', unsafe_allow_html=True)

    with tab_speaker:
        st.markdown("### Speaker Diarization (Simulated)")
        for seg in res["segments"]:
            with st.chat_message(seg["speaker"]):
                st.write(seg["text"])
                st.caption(f"{seg['word_count']} words")

    with tab_sec:
        if res["security"]["has_sensitive_content"]:
            st.error("### 🚨 Sensitive Data Detected")
            for name, info in res["security"]["findings"].items():
                st.write(f"**{name}**: {info['count']} instances found.")
                st.info(f"Masked Examples: {', '.join(info['examples'])}")
        else:
            st.success("✅ No sensitive data patterns (SSN, Email, CC) detected.")

    with tab_biz:
        st.markdown("### 💰 ROI & Business Value")
        saved_mins = res["audio_info"]["duration_seconds"] / 60 * 2 # Estimation: 2x audio duration saved
        st.metric("Estimated Time Saved", f"{saved_mins:.1f} mins")
        
        st.markdown("#### Pricing Tiers")
        pcols = st.columns(len(config.PRICING_TIERS))
        for i, (name, tier) in enumerate(config.PRICING_TIERS.items()):
            with pcols[i]:
                st.markdown(f"""
                <div style="background:{tier['color']}22; border:2px solid {tier['color']}; padding:20px; border-radius:20px; height:350px;">
                    <h3 style="color:{tier['color']};">{name}</h3>
                    <h2>{tier['price']}</h2>
                    <ul style="font-size:0.8rem;">
                        {''.join([f"<li>{f}</li>" for f in tier['features']])}
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    # Export Section
    st.markdown("---")
    st.markdown("### 📤 Export Intelligence")
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        st.download_button("Download Transcript (TXT)", res["transcript"], file_name="transcript.txt")
    with col_ex2:
        minutes = nlp.generate_meeting_minutes(res["transcript"], res["summary"], res["action_items"], res["sentiment"], res["topics"])
        st.download_button("Download Meeting Minutes (MD)", minutes, file_name="minutes.md")

else:
    # Landing Page
    st.markdown("""
    <div style="text-align:center; padding: 5rem 2rem;">
        <h2 style="font-size:2.5rem; margin-bottom:1rem;">Welcome to the Future of Meetings</h2>
        <p style="font-size:1.2rem; color:#94a3b8; max-width:800px; margin:0 auto;">
            Our AI-powered platform transforms raw audio into actionable business intelligence. 
            Upload a recording to see the power of Whisper-Medium and our advanced NLP pipeline.
        </p>
        <div style="margin-top:3rem;">
            <img src="https://img.icons8.com/fluency/240/artificial-intelligence.png" width="120" style="opacity:0.8;">
        </div>
    </div>
    """, unsafe_allow_html=True)
