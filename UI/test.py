import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import sounddevice as sd
import pandas as pd
import time
import soundfile as sf
import io
import requests
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase
import cv2

# Set page configuration
st.set_page_config(
    page_title="Emotion Recognition System",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    .main {
        padding: 2rem;
    }
    .stButton>button {
        border-radius: 20px;
        padding: 0.5rem 2rem;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .emotion-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load pre-trained models
@st.cache_resource
def download_model(url, save_path):
    """Download model if it does not exist."""
    if not os.path.exists(save_path):
        response = requests.get(url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

@st.cache_resource
def load_models():
    """Load the face and audio emotion recognition models."""
    tf.keras.backend.clear_session()

    face_model_url = "https://github.com/Morampudi-Buddu-Uday/EMR-sub-main/raw/refs/heads/sub_main/UI/face_emotion_recognition_model1.h5"
    audio_model_url = "https://github.com/Morampudi-Buddu-Uday/EMR-sub-main/raw/refs/heads/sub_main/UI/Emotion_Voice_Detection_Model1.h5"

    face_model_path = "Models/face_emotion_recognition_model1.h5"
    audio_model_path = "Models/Emotion_Voice_Detection_Model1.h5"

    os.makedirs("Models", exist_ok=True)

    download_model(face_model_url, face_model_path)
    download_model(audio_model_url, audio_model_path)

    try:
        face_model = tf.keras.models.load_model(face_model_path)
        audio_model = tf.keras.models.load_model(audio_model_path)
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        raise

    return face_model, audio_model

# Load models
face_model, audio_model = load_models()

# Emotion labels with emojis
emotions = {
    'Neutral': '😐',
    'Happy': '😊',
    'Sad': '😢',
    'Angry': '😠'
}

# Helper functions
def preprocess_frame(frame, target_size=(128, 128)):
    """Preprocess the captured frame for the face model."""
    frame_resized = cv2.resize(frame, target_size)
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)

def preprocess_audio(audio, sr=22050, n_mfcc=40):
    """Extract MFCC features from audio for model input."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return np.expand_dims(mfccs_mean, axis=0)

def weighted_emotion_prediction(face_probs, audio_probs):
    """Combine face and audio predictions with dynamic weighting."""
    face_weight, audio_weight = 0.5, 0.5  # Equal weight for simplicity
    final_probs = face_weight * face_probs + audio_weight * audio_probs
    final_emotion_index = np.argmax(final_probs)
    return list(emotions.keys())[final_emotion_index], final_probs

# Video Transformer for WebRTC
class VideoTransformer(VideoTransformerBase):
    def _init_(self):
        self.frame = None

    def transform(self, frame):
        self.frame = frame.to_ndarray(format="bgr24")
        st.session_state.captured_face = self.frame
        return self.frame

# App layout
st.title("🎭 Emotion Recognition System")
st.markdown("### Discover your emotion and get the perfect music recommendation!")

# Step 1: Facial Capture
st.markdown("### 📸 Step 1: Facial Capture")
webrtc_ctx = webrtc_streamer(
    key="face_capture",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=VideoTransformer
)

if webrtc_ctx.video_transformer:
    st.session_state.captured_face = webrtc_ctx.video_transformer.frame

# Step 2: Voice Recording
st.markdown("### 🎤 Step 2: Voice Recording")
if st.button("🎙 Record Audio (5s)", key="record_audio"):
    with st.spinner("Recording..."):
        audio_data = []
        duration = 5
        sr = 22050

        def audio_callback(indata, frames, time, status):
            audio_data.append(indata.copy())

        with sd.InputStream(samplerate=sr, channels=1, callback=audio_callback):
            time.sleep(duration)

        audio = np.concatenate(audio_data).flatten()
        st.session_state.audio_input = audio
        st.session_state.audio_sample_rate = sr
        st.success("✅ Audio recorded successfully!")

        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sr, format='WAV')
        wav_buffer.seek(0)
        st.audio(wav_buffer, format="audio/wav")

# Step 3: Emotion Analysis
st.markdown("### 🎯 Step 3: Emotion Analysis")
if st.session_state.captured_face is not None and 'audio_input' in st.session_state:
    if st.button("🔍 Analyze Emotion"):
        with st.spinner("Analyzing..."):
            face_input = preprocess_frame(st.session_state.captured_face)
            audio_input = preprocess_audio(st.session_state.audio_input, sr=st.session_state.audio_sample_rate)

            face_probs = face_model.predict(face_input)
            audio_probs = audio_model.predict(audio_input)

            final_emotion, _ = weighted_emotion_prediction(face_probs, audio_probs)

            st.success(f"🎭 Final Emotion: {emotions[final_emotion]} {final_emotion}")
else:
    st.info("Please capture your face and record your voice before analysis.")

# Step 4: Music Recommendation
st.markdown("### 🎵 Step 4: Music Recommendation")
if 'final_emotion' in st.session_state:
    emotion = st.session_state.get('final_emotion', None)
    if emotion:
        st.markdown(f"Based on your emotion ({emotions[emotion]} {emotion}), here are some music recommendations:")

        # Example music recommendation logic
        music_recommendations = {
            'Neutral': ['Relaxing Instrumentals', 'Lo-fi Beats'],
            'Happy': ['Upbeat Pop Songs', 'Feel-Good Hits'],
            'Sad': ['Mellow Acoustic Tracks', 'Soothing Piano Music'],
            'Angry': ['Rock Anthems', 'Energetic Workout Music']
        }

        recommendations = music_recommendations.get(emotion, [])
        for i, track in enumerate(recommendations, 1):
            st.markdown(f"{i}. {track}")
    else:
        st.info("Please analyze your emotion first to see recommendations.")
else:
    st.info("Analyze your emotion to see personalized music recommendations.")

# Utility Functions
@st.cache
def log_emotion_data(emotion, timestamp):
    """Log detected emotion and timestamp into a CSV file."""
    log_file = "emotion_log.csv"
    entry = {"Timestamp": timestamp, "Emotion": emotion}

    if not os.path.exists(log_file):
        df = pd.DataFrame([entry])
        df.to_csv(log_file, index=False)
    else:
        df = pd.read_csv(log_file)
        df = df.append(entry, ignore_index=True)
        df.to_csv(log_file, index=False)

# Step 5: Log and History
st.markdown("### 📜 Step 5: Emotion History")
if st.button("📝 Log Emotion"):
    if 'final_emotion' in st.session_state:
        emotion = st.session_state['final_emotion']
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_emotion_data(emotion, timestamp)
        st.success(f"Emotion logged: {emotion} at {timestamp}")
    else:
        st.warning("No emotion detected to log. Analyze your emotion first.")

# Display emotion log
if os.path.exists("emotion_log.csv"):
    st.markdown("### 📚 Emotion History Log")
    history_df = pd.read_csv("emotion_log.csv")
    st.dataframe(history_df)

# Conclusion
st.markdown("### 🎉 Thank you for using the Emotion Recognition System!")
st.markdown("Feel free to reach out for feedback or suggestions to improve the system.")
