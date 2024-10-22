import streamlit as st

def app():
    st.title("Welcome to EmotionAI")
    
    st.write(
        """
        EmotionAI is an advanced application designed to detect and analyze emotions from audio inputs. 
        It is an artificial neural network model to provide emotion detection and 
        real-time monitoring for driver anger alerts.
        """
    )
    

    st.write("### Features")
    st.write(
        """
        - **Emotion Detection**: Upload an audio file to detect and analyze emotions. The app supports various emotions such as happiness, sadness, anger, and more.
        - **Driver Anger Alert**: Real-time monitoring to detect anger in drivers. The app alerts the driver when an angry emotion is detected to promote safe driving.
        """
    )
    
    st.write("### How to Use")
    st.write(
        """
        1. **Emotion Detection**: Navigate to the 'Emotion Detection' page to upload an audio file. The app will process the file and display the detected emotion.
        2. **Driver Anger Alert**: Navigate to the 'Driver Anger Alert' page to start real-time monitoring. The app will alert you if it detects anger in your voice.
        """
    )
    
    st.write("### Tech Stack")
    st.write(
        """
        1. **Language**: Python
        2. **Libraries**: Keras, TensorFlow, librosa, soundfile, sklearn, pandas, matplotlib, NumPy, pickle, mlfoundry.
        """
    )
    
