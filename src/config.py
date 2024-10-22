import configparser
import os

config = configparser.RawConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '../input/config.ini'))

MODEL_SAVE_PATH = config.get('MODEL', 'model_save_path')
EMOTIONS_LABEL = eval(config.get('DATA', 'emotions'))
DATA_DIR = config.get('DATA', 'data_dir')