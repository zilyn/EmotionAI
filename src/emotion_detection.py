import streamlit as st
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model
from ml_pipeline.utils import extract_feature
from engine import EMOTIONS_LABEL, config

def app():
    st.title("Emotion Detection")
    
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        
        # Load and preprocess the audio file
        feature = extract_feature(uploaded_file)
        feature = feature.reshape(1, -1)
        
        # Load the pre-trained model
        model_path = config.get('MODEL', 'model_save_path')
        model = load_model(model_path + '/keras')
        
        # Predict the emotion
        predicted_emotion = model.predict(feature)
        predicted_emotion_index = np.argmax(predicted_emotion)
        
        # Map the predicted index to the emotion label
        emotion = EMOTIONS_LABEL['0' + str(predicted_emotion_index + 1)]
        
        st.write(f"Predicted Emotion: {emotion}")