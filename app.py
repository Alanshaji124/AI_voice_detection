"""
===============================================================================
AI-Generated Voice Detection System — Gradio Web Application
===============================================================================
Academic Mini-Project: MCA Machine Learning
Backend: Support Vector Machine (RBF Kernel) with 26 Spectral & MFCC Features
===============================================================================
"""

import os
import joblib
import librosa
import numpy as np
import pandas as pd
import gradio as gr

# =============================================================================
# 1. MODEL & ARTIFACT LOADING
# =============================================================================

MODEL_PATH = "svm_voice_detector.pkl"
SCALER_PATH = "voice_scaler.pkl"
FEATURE_NAMES_PATH = "feature_names.pkl"

model = None
scaler = None
feature_names = None
model_loaded = False
load_error_msg = ""

def load_artifacts():
    global model, scaler, feature_names, model_loaded, load_error_msg
    missing_files = []
    
    if not os.path.exists(MODEL_PATH):
        missing_files.append(MODEL_PATH)
    if not os.path.exists(SCALER_PATH):
        missing_files.append(SCALER_PATH)
    if not os.path.exists(FEATURE_NAMES_PATH):
        missing_files.append(FEATURE_NAMES_PATH)
        
    if missing_files:
        load_error_msg = f"Missing required model file(s): {', '.join(missing_files)}. Please place them in the project folder."
        model_loaded = False
        return
        
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        model_loaded = True
        print(f"[SUCCESS] Loaded Model ({MODEL_PATH}), Scaler ({SCALER_PATH}), and Features ({FEATURE_NAMES_PATH}).")
    except Exception as e:
        load_error_msg = f"Error loading model artifacts: {str(e)}"
        model_loaded = False

# Attempt initial load
load_artifacts()


# =============================================================================
# 2. FEATURE EXTRACTION PIPELINE (26 Features per 1-second segment)
# =============================================================================

def extract_segment_features(segment, sr):
    """
    Extracts exactly 26 acoustic features for a 1-second audio segment:
    1. chroma_stft (mean)
    2. rms (mean)
    3. spectral_centroid (mean)
    4. spectral_bandwidth (mean)
    5. rolloff (mean)
    6. zero_crossing_rate (mean)
    7-26. mfcc1 through mfcc20 (means)
    """
    # 1. Chroma STFT
    chroma_stft = float(np.mean(librosa.feature.chroma_stft(y=segment, sr=sr)))
    
    # 2. RMS Energy
    rms = float(np.mean(librosa.feature.rms(y=segment)))
    
    # 3. Spectral Centroid
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr)))
    
    # 4. Spectral Bandwidth
    spectral_bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr)))
    
    # 5. Spectral Rolloff
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=segment, sr=sr)))
    
    # 6. Zero Crossing Rate
    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y=segment)))
    
    # 7-26. MFCCs 1 to 20
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20)
    mfcc_means = [float(np.mean(mfccs[i])) for i in range(20)]
    
    feature_vector = [
        chroma_stft,
        rms,
        spectral_centroid,
        spectral_bandwidth,
        rolloff,
        zero_crossing_rate
    ] + mfcc_means
    
    return feature_vector


# =============================================================================
# 3. AUDIO PREDICTION LOGIC
# =============================================================================

def analyze_audio(audio_path):
    """
    Processes uploaded audio file, segments into 1-second clips, extracts features,
    scales using saved scaler, predicts using SVM, and computes majority vote.
    """
    # 1. Check model status
    if not model_loaded:
        # Retry loading in case user placed files after app start
        load_artifacts()
        if not model_loaded:
            return (
                "⚠️ **MODEL ERROR**",
                f"<div style='color:red; font-weight:bold;'>{load_error_msg}</div>",
                "-", "-", "-", "-", "-", "-"
            )

    # 2. Handle missing audio file
    if audio_path is None or not os.path.exists(audio_path):
        return (
            "⚠️ **INPUT ERROR**",
            "<div style='color:orange;'>Please upload or record an MP3 or WAV audio file first.</div>",
            "-", "-", "-", "-", "-", "-"
        )

    try:
        # 3. Load audio with original sample rate & convert to mono
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        return (
            "❌ **AUDIO PROCESSING ERROR**",
            f"<div>Could not read audio file. Format may be unsupported or file corrupt: {str(e)}</div>",
            "-", "-", "-", "-", "-", "-"
        )

    # 4. Limit analysis to maximum first 30 seconds
    max_samples = sr * 30
    if len(y) > max_samples:
        y = y[:max_samples]

    # 5. Verify duration (minimum 1 second required)
    total_samples = len(y)
    duration_sec = total_samples / sr

    if duration_sec < 1.0:
        return (
            "⚠️ **AUDIO TOO SHORT**",
            f"<div>Audio duration is {duration_sec:.2f} seconds. Audio must be at least 1.0 second long for segment analysis.</div>",
            "-", "-", "-", "-", "-", "-"
        )

    # 6. Divide audio into non-overlapping 1-second segments
    segment_samples = int(sr * 1.0)
    num_segments = int(total_samples // segment_samples)

    if num_segments == 0:
        return (
            "⚠️ **INSUFFICIENT DATA**",
            "<div>Audio clip is too short to form a full 1-second segment.</div>",
            "-", "-", "-", "-", "-", "-"
        )

    # 7. Segment-level feature extraction
    segment_features = []
    for i in range(num_segments):
        start_idx = i * segment_samples
        end_idx = start_idx + segment_samples
        seg = y[start_idx:end_idx]
        
        feats = extract_segment_features(seg, sr)
        segment_features.append(feats)

    # Convert to 2D numpy array
    X_raw = np.array(segment_features)

    # Ensure feature names alignment if scaler expects DataFrame column names
    default_names = [
        'chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth',
        'rolloff', 'zero_crossing_rate'
    ] + [f'mfcc{i}' for i in range(1, 21)]

    cols_to_use = feature_names if feature_names is not None else default_names

    try:
        if hasattr(scaler, 'feature_names_in_'):
            X_df = pd.DataFrame(X_raw, columns=scaler.feature_names_in_)
            X_scaled = scaler.transform(X_df)
        else:
            X_scaled = scaler.transform(X_raw)
    except Exception:
        # Fallback to direct array transform
        X_scaled = scaler.transform(X_raw)

    # 8. Predict per segment
    segment_preds = model.predict(X_scaled)          # 0 = REAL, 1 = FAKE
    segment_probas = model.predict_proba(X_scaled)    # Probabilities for [Class 0, Class 1]

    real_count = int(np.sum(segment_preds == 0))
    fake_count = int(np.sum(segment_preds == 1))

    real_pct = (real_count / num_segments) * 100.0
    fake_pct = (fake_count / num_segments) * 100.0

    avg_real_score = float(np.mean(segment_probas[:, 0])) * 100.0
    avg_fake_score = float(np.mean(segment_probas[:, 1])) * 100.0

    # 9. Majority voting with tie-breaker
    if fake_count > real_count:
        final_prediction = "FAKE / AI-GENERATED VOICE"
        result_badge = """
        <div style='background-color: #fee2e2; border: 2px solid #ef4444; padding: 16px; border-radius: 8px; text-align: center;'>
            <h2 style='margin:0; font-size: 24px; color: #000000 !important; font-weight: bold;'>🤖 FAKE / AI-GENERATED VOICE</h2>
            <p style='margin:6px 0 0 0; font-size: 14px; color: #111827 !important; font-weight: 500;'>Synthetic voice patterns detected across segments.</p>
        </div>
        """
    elif real_count > fake_count:
        final_prediction = "REAL HUMAN VOICE"
        result_badge = """
        <div style='background-color: #dcfce7; border: 2px solid #22c55e; padding: 16px; border-radius: 8px; text-align: center;'>
            <h2 style='margin:0; font-size: 24px; color: #000000 !important; font-weight: bold;'>👤 REAL HUMAN VOICE</h2>
            <p style='margin:6px 0 0 0; font-size: 14px; color: #111827 !important; font-weight: 500;'>Natural human acoustic characteristics confirmed.</p>
        </div>
        """
    else:
        # Tie break using average probability
        if avg_fake_score > avg_real_score:
            final_prediction = "FAKE / AI-GENERATED VOICE"
            result_badge = """
            <div style='background-color: #fee2e2; border: 2px solid #ef4444; padding: 16px; border-radius: 8px; text-align: center;'>
                <h2 style='margin:0; font-size: 24px; color: #000000 !important; font-weight: bold;'>🤖 FAKE / AI-GENERATED VOICE</h2>
                <p style='margin:6px 0 0 0; font-size: 14px; color: #111827 !important; font-weight: 500;'>Tied segments — Decided by higher average synthetic score.</p>
            </div>
            """
        else:
            final_prediction = "REAL HUMAN VOICE"
            result_badge = """
            <div style='background-color: #dcfce7; border: 2px solid #22c55e; padding: 16px; border-radius: 8px; text-align: center;'>
                <h2 style='margin:0; font-size: 24px; color: #000000 !important; font-weight: bold;'>👤 REAL HUMAN VOICE</h2>
                <p style='margin:6px 0 0 0; font-size: 14px; color: #111827 !important; font-weight: 500;'>Tied segments — Decided by higher average human score.</p>
            </div>
            """

    return (
        final_prediction,
        result_badge,
        f"{num_segments} (analyzed first {duration_sec:.1f}s)",
        f"{real_count} ({real_pct:.1f}%)",
        f"{fake_count} ({fake_pct:.1f}%)",
        f"{avg_real_score:.2f}%",
        f"{avg_fake_score:.2f}%",
        "Analysis Complete"
    )


# =============================================================================
# 4. GRADIO UI DESIGN
# =============================================================================

custom_css = """
.container { max-width: 1000px; margin: auto; padding: 10px; }
.title-box { text-align: center; margin-bottom: 15px; }
.info-card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-top: 10px; }
"""

with gr.Blocks(title="AI-Generated Voice Detection System", css=custom_css) as demo:
    
    # Title & Subtitle Header
    gr.Markdown(
        """
        # 🎙️ AI-Generated Voice Detection System
        ### Detect whether an uploaded voice recording is real human speech or AI-generated.
        ---
        """
    )

    with gr.Row():
        # LEFT COLUMN — Inputs
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Upload Voice Sample (MP3 or WAV)",
                type="filepath",
                sources=["upload", "microphone"]
            )
            
            analyze_btn = gr.Button("🔍 Analyze Voice", variant="primary", size="lg")
            
            gr.Markdown(
                """
                ### 📋 Processing Details
                * **Format Support:** MP3, WAV
                * **Analysis Limit:** First 30 seconds
                * **Window Size:** 1-second non-overlapping segments
                * **Feature Count:** 26 features per segment
                """
            )

        # RIGHT COLUMN — Results & Analysis
        with gr.Column(scale=1):
            # Output Badge
            result_header = gr.Markdown("### 📊 Classification Result")
            output_badge = gr.HTML(
                "<div style='background-color: #f1f5f9; border: 1px solid #cbd5e1; padding: 16px; border-radius: 8px; text-align: center;'><p style='margin:0; color: #111827 !important; font-size: 15px; font-weight: 500;'>Upload an audio file and click <b style='color: #000000 !important;'>Analyze Voice</b> to see predictions.</p></div>"
            )
            
            # Segment Statistics
            gr.Markdown("### ⏱️ Segment Voting Analysis")
            with gr.Row():
                num_segments_out = gr.Textbox(label="Total Segments Analyzed", interactive=False)
                status_out = gr.Textbox(label="Status", interactive=False)

            with gr.Row():
                real_segments_out = gr.Textbox(label="REAL Segments", interactive=False)
                fake_segments_out = gr.Textbox(label="FAKE Segments", interactive=False)

            # Model Scores
            gr.Markdown("### 📈 Average Model Scores")
            with gr.Row():
                real_score_out = gr.Textbox(label="Average REAL Model Score", interactive=False)
                fake_score_out = gr.Textbox(label="Average FAKE Model Score", interactive=False)

    # Disclaimer Note
    gr.Markdown(
        """
        > 💡 **Model Score Note:**
        > Predictions are based on acoustic patterns learned from the training dataset and may be less reliable for audio from different recording conditions, speakers, codecs, or generation methods. Model scores represent classifier decision margins on 1-second segments rather than guaranteed real-world authenticity probability.
        """
    )

    # Model Information Section
    with gr.Accordion("ℹ️ Model Information & Benchmark Evaluation", open=False):
        gr.Markdown(
            r"""
            | Parameter | Description |
            |---|---|
            | **Classification Model** | Support Vector Machine (RBF Kernel) |
            | **Input Features** | 26 (1 Chroma, 1 RMS, 1 Centroid, 1 Bandwidth, 1 Rolloff, 1 ZCR, 20 MFCCs) |
            | **Segment Duration** | 1-Second Audio Segments |
            | **Target Classes** | 0 = REAL HUMAN VOICE | 1 = FAKE / AI-GENERATED VOICE |
            | **Scaler Used** | Saved `StandardScaler` (Fit strictly on training set) |
            
            #### 📊 Benchmark Test Performance (Held-out Test Set)
            * **Accuracy:** 99.32%
            * **Precision:** 99.21%
            * **Recall:** 99.43%
            * **F1-Score:** 99.32%
            * **ROC-AUC:** 99.98%
            """
        )

    # Hidden text output for simple programmatic access if needed
    prediction_text_hidden = gr.Textbox(visible=False)

    # Button Event Binding
    analyze_btn.click(
        fn=analyze_audio,
        inputs=[audio_input],
        outputs=[
            prediction_text_hidden,
            output_badge,
            num_segments_out,
            real_segments_out,
            fake_segments_out,
            real_score_out,
            fake_score_out,
            status_out
        ]
    )


# =============================================================================
# 5. APPLICATION LAUNCH CODE
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Starting AI-Generated Voice Detection System Gradio Interface...")
    print("=" * 70)
    demo.launch(share=False)
