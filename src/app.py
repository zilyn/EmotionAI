import streamlit as st
import home
import emotion_detection
import driver_anger_alert
import os

# Load configuration
config = configparser.RawConfigParser()
config_file_path = os.path.join(os.path.dirname(__file__), '../input/config.ini')

# Debugging information
st.write(f"Config file path: {config_file_path}")
if not os.path.exists(config_file_path):
    st.error(f"Config file not found at {config_file_path}")

config.read(config_file_path)

# Debugging information
st.write(f"Sections in config file: {config.sections()}")

MODEL_SAVE_PATH = config.get('MODEL', 'model_save_path')
EMOTIONS_LABEL = eval(config.get('DATA', 'emotions'))

# Custom CSS
st.markdown(
    """
    <style>

    /* Position the logo at the top left corner */
    .logo {
        position: absolute;
        width: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
logo_path = os.path.join(os.path.dirname(__file__), '../i.png')
st.sidebar.image(logo_path, width=130)
st.sidebar.title("Choose App")
page = st.sidebar.radio("Go to", ["🏠 Home", "😊 Emotion Detection", "🚗 Driver Anger Alert"])

# Navigation logic
if page == "🏠 Home":
    # Display logo at the top left corner only on the homepage
    logo_path = os.path.join(os.path.dirname(__file__), '../i.png')
    st.image(logo_path, width=130)
    home.app()
elif page == "😊 Emotion Detection":
    emotion_detection.app()
elif page == "🚗 Driver Anger Alert":
    driver_anger_alert.app()
