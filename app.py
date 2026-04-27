"""
AI Speech Intelligence System - Main Streamlit Application
Smart Meeting & Lecture Assistant powered by openai/whisper-small
"""

import streamlit as st
import numpy as np
import time
import plotly.graph_objects as go
import plotly.express as px

# -- Page Config --
st.set_page_config(
    page_title="AI Speech Intelligence System",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- Custom CSS --
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 16px; margin-bottom: 2rem;
    color: white; text-align: center;
}
.main-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
.main-header p { opacity: 0.9; font-size: 1.1rem; }
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    padding: 1.2rem; border-radius: 12px; border: 1px solid #334155;
    text-align: center; color: white;
}
.metric-card h3 { font-size: 0.85rem; color: #94a3b8; margin: 0 0 0.3rem 0; }
.metric-card p { font-size: 1.5rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.result-box {
    background: #0f172a; border: 1px solid #334155; border-radius: 12px;
    padding: 1.5rem; margin: 1rem 0; color: #e2e8f0;
}
.action-item { background: #1e293b; border-left: 3px solid #667eea;
    padding: 0.6rem 1rem; margin: 0.4rem 0; border-radius: 0 8px 8px 0; color: #e2e8f0; }
.sentiment-pos { color: #4ade80; font-weight: 600; }
.sentiment-neg { color: #f87171; font-weight: 600; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #1e293b; border-radius: 8px; color: #94a3b8;
    padding: 8px 20px; border: 1px solid #334155;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Model Loading (cached)
# ------------------------------------------------------------------ #
@st.cache_resource
def load_whisper():
    from models.whisper_model import WhisperASR
    model = WhisperASR()
    model.load_model()
    return model

@st.cache_resource
def load_nlp(device, quick_mode=True):
    from models.nlp_pipeline import NLPPipeline
    return NLPPipeline(device=device, quick_mode=quick_mode)

def load_evaluator():
    from models.evaluation import TranscriptionEvaluator
    return TranscriptionEvaluator()


# ------------------------------------------------------------------ #
#  Header
# ------------------------------------------------------------------ #
st.markdown("""
<div class="main-header">
    <h1>🎙️ AI Speech Intelligence System</h1>
    <p>GPU-Accelerated Meeting & Lecture Assistant · Powered by OpenAI Whisper-small</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  Sidebar
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    language = st.selectbox("Source Language", [
        "Auto Detect", "English", "Spanish", "French", "German",
        "Chinese", "Japanese", "Arabic", "Hindi", "Portuguese", "Russian"
    ])
    lang_code_map = {
        "Auto Detect": None, "English": "en", "Spanish": "es",
        "French": "fr", "German": "de", "Chinese": "zh",
        "Japanese": "ja", "Arabic": "ar", "Hindi": "hi",
        "Portuguese": "pt", "Russian": "ru",
    }
    lang_code = lang_code_map[language]

    task = st.radio("Whisper Task", ["Transcribe", "Translate to English"])
    whisper_task = "transcribe" if task == "Transcribe" else "translate"

    st.markdown("---")
    st.markdown("### 📊 Evaluation Settings")
    run_eval = st.checkbox("Run Baseline vs Improved Comparison", value=False)
    ref_text = st.text_area("Reference Text (for WER)", "", height=100,
                            help="Paste ground-truth text to compute WER")

    st.markdown("---")
    st.markdown("### 🔊 Noise Testing")
    from utils.noise_augmentation import NOISE_PRESETS
    noise_preset = st.selectbox("Noise Preset", list(NOISE_PRESETS.keys()))

    st.markdown("---")
    st.markdown("### 🌐 Translation")
    translate_output = st.checkbox("Translate Output", value=False)
    target_lang = st.selectbox("Target Language", [
        "Spanish", "French", "German", "Chinese",
        "Japanese", "Arabic", "Hindi", "Portuguese", "Russian"
    ])

    st.markdown("---")
    st.markdown("### ⚡ Performance")
    quick_mode = st.checkbox(
        "Quick Mode (faster NLP outputs)",
        value=True,
        help="Uses faster summary/analysis settings for quick results."
    )

    st.markdown("---")
    # Device info
    st.markdown("### 💻 System Info")
    try:
        whisper = load_whisper()
        info = whisper.get_device_info()
        st.success(f"Device: **{info['device'].upper()}**")
        if "gpu_name" in info:
            st.info(f"GPU: {info['gpu_name']}")
    except Exception as e:
        st.error(f"Model loading issue: {e}")

# ------------------------------------------------------------------ #
#  Main Area - Audio Input
# ------------------------------------------------------------------ #
st.markdown("## 🎧 Audio Input")
input_tabs = st.tabs(["📁 Upload File", "🎤 Record from Microphone"])

uploaded = None
mic_audio = None

with input_tabs[0]:
    uploaded = st.file_uploader(
        "Upload an audio file (WAV, MP3, M4A, FLAC, OGG)",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Whisper works best with clear speech audio at 16kHz"
    )

with input_tabs[1]:
    if hasattr(st, "audio_input"):
        mic_audio = st.audio_input("Record live speech and click stop")
        st.caption("After recording, click 'Run Speech Intelligence Pipeline'.")
    else:
        st.warning("Your Streamlit version does not support microphone input (`st.audio_input`).")
        st.info("Use file upload mode or upgrade Streamlit.")

selected_audio = uploaded if uploaded is not None else mic_audio

if selected_audio is not None:
    if uploaded is not None:
        st.audio(uploaded, format=f"audio/{uploaded.name.split('.')[-1]}")
    else:
        st.audio(mic_audio)

    # Load and process audio
    from utils.audio_utils import load_audio, get_audio_info
    from utils.noise_augmentation import apply_noise_preset

    with st.spinner("🔄 Loading audio..."):
        audio_bytes = selected_audio.read()
        audio, sr = load_audio(audio_bytes)
        audio_info = get_audio_info(audio, sr)

    # Display audio metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>Duration</h3><p>{audio_info["duration_seconds"]}s</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>Sample Rate</h3><p>{audio_info["sampling_rate"]} Hz</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>RMS Energy</h3><p>{audio_info["rms_energy"]}</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>Peak</h3><p>{audio_info["peak_amplitude"]}</p></div>', unsafe_allow_html=True)

    # Apply noise if selected
    processed_audio = apply_noise_preset(audio, noise_preset)

    # ------------------------------------------------------------------ #
    #  Transcription
    # ------------------------------------------------------------------ #
    if st.button("🚀 Run Speech Intelligence Pipeline", type="primary", use_container_width=True):
        try:
            whisper = load_whisper()
            nlp = load_nlp(whisper.device, quick_mode=quick_mode)
            evaluator = load_evaluator()
        except Exception as e:
            st.error(
                "Model loading failed. This is usually caused by blocked Hugging Face access "
                "or missing local model cache. Try again after disabling proxy or pre-downloading the model."
            )
            st.exception(e)
            st.stop()

        # -- Step 1: Transcribe --
        with st.spinner("🎤 Transcribing with Whisper-small..."):
            result = whisper.transcribe(
                processed_audio, sr, language=lang_code, task=whisper_task
            )

        transcript = result["text"]

        # Store in session state
        st.session_state["transcript"] = transcript
        st.session_state["result"] = result

        # -- Tabs for results --
        tabs = st.tabs(["📝 Transcript", "📋 Summary", "✅ Action Items",
                        "💭 Sentiment", "🏷️ Topics", "🌐 Translation", "📊 Evaluation"])

        # Tab 1: Transcript
        with tabs[0]:
            st.markdown("### Transcription Result")
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Latency", f"{result['latency_seconds']}s")
            col_b.metric("Device", result["device"].upper())
            col_c.metric("Language", result["language"])
            st.markdown(f'<div class="result-box">{transcript}</div>', unsafe_allow_html=True)

        # Tab 2: Summary
        with tabs[1]:
            st.markdown("### AI Summary")
            if len(transcript.split()) > 20:
                with st.spinner("Generating summary..."):
                    summary = nlp.summarize(transcript)
                st.markdown(f'<div class="result-box">{summary}</div>', unsafe_allow_html=True)
            else:
                st.info("Text too short for summarization.")

        # Tab 3: Action Items
        with tabs[2]:
            st.markdown("### Extracted Action Items")
            with st.spinner("Extracting action items..."):
                items = nlp.extract_action_items(transcript)
            for i, item in enumerate(items, 1):
                st.markdown(f'<div class="action-item">📌 <strong>{i}.</strong> {item}</div>', unsafe_allow_html=True)

        # Tab 4: Sentiment
        with tabs[3]:
            st.markdown("### Sentiment Analysis")
            with st.spinner("Analyzing sentiment..."):
                sent = nlp.analyze_sentiment(transcript)

            overall = sent["overall"]
            cls = "sentiment-pos" if overall["label"] == "POSITIVE" else "sentiment-neg"
            st.markdown(f'Overall: <span class="{cls}">{overall["label"]} ({overall["score"]:.1%})</span>', unsafe_allow_html=True)

            bd = sent["breakdown"]
            fig = go.Figure(data=[go.Pie(
                labels=["Positive", "Negative"],
                values=[bd["positive_pct"], bd["negative_pct"]],
                marker_colors=["#4ade80", "#f87171"],
                hole=0.5,
            )])
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Sentence-level Details"):
                for d in sent["details"]:
                    c = "🟢" if d["label"] == "POSITIVE" else "🔴"
                    st.markdown(f"{c} **{d['label']}** ({d['score']:.1%}): {d['sentence']}")

        # Tab 5: Topics
        with tabs[4]:
            st.markdown("### Key Topics")
            topics = nlp.extract_key_topics(transcript)
            if topics:
                words, counts = zip(*topics)
                fig = px.bar(x=list(words), y=list(counts),
                             labels={"x": "Keyword", "y": "Frequency"},
                             color=list(counts),
                             color_continuous_scale="Viridis")
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Tab 6: Translation
        with tabs[5]:
            st.markdown(f"### Translation to {target_lang}")
            if translate_output:
                with st.spinner(f"Translating to {target_lang}..."):
                    translated = nlp.translate(transcript, target_lang)
                st.markdown(f'<div class="result-box">{translated}</div>', unsafe_allow_html=True)
            else:
                st.info("Enable 'Translate Output' in the sidebar to use this feature.")

        # Tab 7: Evaluation
        with tabs[6]:
            st.markdown("### Evaluation & Benchmarking")

            if run_eval:
                with st.spinner("⏱️ Running evaluation (baseline vs improved)..."):
                    ref = ref_text.strip() if ref_text.strip() else None
                    eval_results = evaluator.quick_evaluate(
                        whisper, processed_audio, sr,
                        reference_text=ref, language=lang_code
                    )

                for name, data in eval_results.items():
                    with st.expander(f"📐 {name}", expanded=True):
                        st.text(f"Transcription: {data['text'][:300]}...")
                        st.metric("Latency", f"{data['latency_seconds']}s")
                        if "wer_metrics" in data and data["wer_metrics"].get("wer") is not None:
                            wcol1, wcol2, wcol3 = st.columns(3)
                            wcol1.metric("WER", f"{data['wer_metrics']['wer']:.2%}")
                            wcol2.metric("MER", f"{data['wer_metrics']['mer']:.2%}")
                            wcol3.metric("WIL", f"{data['wer_metrics']['wil']:.2%}")

                # Latency comparison chart
                names = list(eval_results.keys())
                latencies = [eval_results[n]["latency_seconds"] for n in names]
                fig = go.Figure(data=[go.Bar(
                    x=names, y=latencies,
                    marker_color=["#667eea", "#764ba2"],
                    text=[f"{l:.3f}s" for l in latencies],
                    textposition="auto",
                )])
                fig.update_layout(
                    title="Latency Comparison",
                    yaxis_title="Seconds",
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=350,
                )
                st.plotly_chart(fig, use_container_width=True)

                # WER comparison if reference provided
                if ref_text.strip():
                    wer_vals = []
                    for n in names:
                        w = eval_results[n].get("wer_metrics", {}).get("wer")
                        wer_vals.append(w if w is not None else 0)
                    fig2 = go.Figure(data=[go.Bar(
                        x=names, y=wer_vals,
                        marker_color=["#4ade80", "#f87171"],
                        text=[f"{w:.2%}" for w in wer_vals],
                        textposition="auto",
                    )])
                    fig2.update_layout(
                        title="WER Comparison (Lower is Better)",
                        yaxis_title="Word Error Rate",
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        height=350,
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Enable 'Run Baseline vs Improved Comparison' in sidebar.")

            # Noise comparison section
            st.markdown("---")
            st.markdown("### 🔊 Noise Impact Analysis")
            if st.button("Run Noise Comparison Test"):
                noise_results = {}
                progress = st.progress(0)
                presets = list(NOISE_PRESETS.keys())

                for i, preset in enumerate(presets):
                    noisy = apply_noise_preset(audio, preset)
                    r = whisper.transcribe(noisy, sr, language=lang_code, task=whisper_task)
                    noise_results[preset] = r
                    progress.progress((i + 1) / len(presets))

                for preset, r in noise_results.items():
                    with st.expander(f"🔊 {preset}"):
                        st.metric("Latency", f"{r['latency_seconds']}s")
                        st.markdown(f'<div class="result-box">{r["text"][:500]}</div>', unsafe_allow_html=True)
                        if ref_text.strip():
                            wm = evaluator.compute_wer(ref_text.strip(), r["text"])
                            if wm.get("wer") is not None:
                                st.metric("WER", f"{wm['wer']:.2%}")

else:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #94a3b8;">
        <h2 style="color: #667eea;">Upload audio or record from microphone</h2>
        <p style="font-size: 1.1rem;">Upload: WAV, MP3, M4A, FLAC, OGG | Mic: live recording</p>
        <p>This system uses <strong>openai/whisper-small</strong> for GPU-accelerated speech recognition,
        then applies NLP models for summarization, sentiment analysis, action items, and translation.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏗️ Pipeline Architecture")
    st.markdown("""
    ```
    Audio Input → Whisper ASR (GPU) → Raw Transcript
                                          ↓
                    ┌─────────────────────┼──────────────────────┐
                    ↓                     ↓                      ↓
               Summarization      Sentiment Analysis       Translation
               Action Items       Key Topics               WER Evaluation
    ```
    """)
