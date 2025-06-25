import numpy as np
import pandas as pd
import librosa
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Class labels (ensure this matches your model's training order)
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

def main():
    model = load_model('crnn_model2.h5')
    df = pd.read_csv('custom_test.csv')  # Your custom dataset CSV

    y_true = []
    y_pred = []

    for idx, row in df.iterrows():
        file_path = row['filepath']
        true_label = int(row['label'])

        try:
            features = preprocess_audio(file_path)
            prediction = model.predict(features, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

        y_true.append(true_label)
        y_pred.append(predicted_class)

        print(f"{file_path}: True={class_labels[true_label]}, Predicted={class_labels[predicted_class]}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_labels))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
