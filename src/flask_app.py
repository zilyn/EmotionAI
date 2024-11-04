from flask import Flask, render_template, request, redirect, url_for
import os
import configparser
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model
import librosa

app = Flask(__name__, template_folder='../templates')

# Load configuration
config = configparser.RawConfigParser()
config_file_path = os.path.join(os.path.dirname(__file__), '../input/config.ini')
config.read(config_file_path)

# Debugging information
print(f"Config file path: {config_file_path}")
if not os.path.exists(config_file_path):
    print(f"Config file not found at {config_file_path}")

print(f"Sections in config file: {config.sections()}")

MODEL_SAVE_PATH = config.get('MODEL', 'model_save_path')
EMOTIONS_LABEL = eval(config.get('DATA', 'emotions'))

def extract_feature(file_name, mfcc=True, chroma=True, mel=True, zero_crossing=True):
    with sf.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if zero_crossing:
            zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
            result = np.hstack((result, zero_crossing))
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emotion_detection', methods=['GET', 'POST'])
def emotion_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            feature = extract_feature(file_path)
            feature = feature.reshape(1, -1)
            model = load_model(os.path.join(MODEL_SAVE_PATH, 'keras'))
            predicted_emotion = model.predict(feature)
            predicted_emotion_index = np.argmax(predicted_emotion)
            emotion = EMOTIONS_LABEL['0' + str(predicted_emotion_index + 1)]
            return render_template('emotion_detection.html', emotion=emotion)
    return render_template('emotion_detection.html')

@app.route('/driver_anger_alert', methods=['GET', 'POST'])
def driver_anger_alert():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            feature = extract_feature(file_path)
            feature = feature.reshape(1, -1)
            model = load_model(os.path.join(MODEL_SAVE_PATH, 'keras'))
            predicted_emotion = model.predict(feature)
            predicted_emotion_index = np.argmax(predicted_emotion)
            emotion = EMOTIONS_LABEL['0' + str(predicted_emotion_index + 1)]
            if emotion == 'angry':
                alert = "Alert: Angry emotion detected!"
            else:
                alert = f"Anger not detected. Detected Emotion: {emotion}"
            return render_template('driver_anger_alert.html', alert=alert)
    return render_template('driver_anger_alert.html')

if __name__ == '__main__':
    app.run(debug=True)