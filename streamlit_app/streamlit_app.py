import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
from io import BytesIO
# No need for pydub or tempfile with this improved approach
import os # <--- Import the os module

# --- No changes needed in this section ---
emotion_map = {
    0: "neutral", 1: "calm", 2: "happy", 3: "sad",
    4: "angry", 5: "fearful", 6: "disgust", 7: "surprised"
}

def preprocess_audio(y, sr, n_mels=128, hop_length=512, n_fft=2048, max_len=100):
    # ... (your existing preprocess_audio function is fine)
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

# --- THIS IS THE UPDATED FUNCTION ---
@st.cache_resource
def load_crnn_model():
    # Construct the absolute path to the model file
    # __file__ is the path to the current script (streamlit_app.py)
    # os.path.dirname gets the directory of the script
    # os.path.join combines it with the model filename
    model_path = os.path.join(os.path.dirname(__file__), 'crnn_model.h5')
    return load_model(model_path)

st.title("Speech Emotion Recognition")

# --- Rest of your code can be simplified ---
uploaded_file = st.file_uploader("Upload a WAV or MP3 file", type=["wav", "mp3"])

if uploaded_file is not None:
    # librosa can read directly from the BytesIO object from the uploader
    y, sr = librosa.load(BytesIO(uploaded_file.read()), sr=None)
    st.audio(uploaded_file, format=uploaded_file.type)

    features = preprocess_audio(y, sr)
    model = load_crnn_model()
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_emotion = emotion_map.get(predicted_class, "Unknown")
    st.success(f"Predicted emotion: {predicted_emotion}")
