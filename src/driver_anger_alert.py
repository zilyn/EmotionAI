import streamlit as st
import numpy as np
import tempfile
import wave
from streamlit_mic_recorder import mic_recorder
from tensorflow.keras.models import load_model
from ml_pipeline.utils import extract_feature
from engine import EMOTIONS_LABEL, MODEL_SAVE_PATH, config

def predict_alert(model, file_path, emotions_label):
    features = extract_feature(file_path)
    data_x = features.reshape(1, -1)
    predicted_emotion = model.predict(data_x)
    predicted_emotion_index = np.argmax(predicted_emotion) + 1
    emotion = emotions_label['0' + str(predicted_emotion_index)]
    return emotion

def save_audio_file(audio_bytes, file_path):
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(44100)  # Sample rate
        wf.writeframes(audio_bytes)

def app():
    st.title("Driver Anger Alert")

    # Initialize session state for audio visibility
    if 'show_audio' not in st.session_state:
        st.session_state.show_audio = False
    if 'audio_bytes' not in st.session_state:
        st.session_state.audio_bytes = None

    # Audio recorder
    st.write("Record your audio:")
    audio_data = mic_recorder(
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=False,
        use_container_width=False,
        callback=None,
        args=(),
        kwargs={},
        key=None
    )

    # Check if audio_data is captured
    if audio_data is not None:
        st.session_state.audio_bytes = audio_data.get("bytes")
        if st.session_state.audio_bytes:
            st.write("Audio recorded successfully.")
            st.session_state.show_audio = True

            # Save the recorded audio to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                save_audio_file(st.session_state.audio_bytes, temp_file.name)
                file_path = temp_file.name

            # Predict the alert
            try:
                model = load_model(MODEL_SAVE_PATH + '/keras')
                alert = predict_alert(model, file_path, EMOTIONS_LABEL)
                if alert == 'angry':
                    st.write("Alert: Angry emotion detected!")
                else:
                    st.write(f"Anger not detected. Detected Emotion: {alert}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    # Display the recorded audio
    if st.session_state.show_audio and st.session_state.audio_bytes:
        st.audio(st.session_state.audio_bytes, format='audio/wav')

# Run the app
if __name__ == "__main__":
    app()
