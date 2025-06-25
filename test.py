import numpy as np
import librosa
from keras.models import load_model

# Load model
model = load_model("crnn_model.h5")
class_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def preprocess_audio(file_path, n_mels=128, hop_length=512, n_fft=2048, max_len=100):
    y, sr = librosa.load(file_path, sr=None)
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

# Example usage
if __name__ == "__main__":
    filepath = "Audio_Song_Actors_01-24/Actor_01/03-02-02-01-01-01-01.wav"
    features = preprocess_audio(filepath)
    prediction = model.predict(features)
    print("Predicted Emotion:", class_labels[np.argmax(prediction)])