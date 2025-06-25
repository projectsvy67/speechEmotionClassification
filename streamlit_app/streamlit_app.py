import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
from io import BytesIO
from pydub import AudioSegment
import tempfile

emotion_map = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

def preprocess_audio(y, sr, n_mels=128, hop_length=512, n_fft=2048, max_len=100):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    if log_mel.shape[1] < max_len:
        pad_width = max_len - log_mel.shape[1]
        log_mel = np.pad(log_mel, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :max_len]
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-9)
    log_mel = np.expand_dims(log_mel, axis=-1)
    log_mel = np.expand_dims(log_mel, axis=0)
    return log_mel

@st.cache_resource
def load_crnn_model():
    return load_model('crnn_model.h5')

st.title("Speech Emotion Recognition")

# File uploader
uploaded_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])

# Audio recorder (built-in)
audio_recorded = st.audio_input("Or record your voice")

audio_bytes = None
audio_format = None

if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    audio_format = uploaded_file.type
elif audio_recorded is not None:
    audio_bytes = audio_recorded.read()
    audio_format = "audio/wav"

if audio_bytes:
    # Save the bytes to a temporary file for librosa
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_bytes)
        file_path = temp_file.name

    y, sr = librosa.load(file_path, sr=None)
    st.audio(audio_bytes, format=audio_format)

    features = preprocess_audio(y, sr)
    model = load_crnn_model()
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_emotion = emotion_map.get(predicted_class, "Unknown")
    st.success(f"Predicted emotion: {predicted_emotion}")
