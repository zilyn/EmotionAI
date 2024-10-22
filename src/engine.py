import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

import argparse
import configparser
import pickle

from tensorflow.keras.models import load_model
import numpy as np

from ml_pipeline.utils import load_train_data, load_infer_data
from ml_pipeline.model import get_model, train

config = configparser.RawConfigParser()
config.read('../input/config.ini')

MODEL_SAVE_PATH    = config.get('MODEL', 'model_save_path')
EMOTIONS_LABEL     = eval(config.get('DATA', 'emotions'))

if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Speech Emotion Detection')
    parser.add_argument('--framework', type=str, default='keras', help='Keras/sklearn')
    parser.add_argument('--train', action='store_true', help='Train')
    parser.add_argument('--infer', action='store_true', help='Inference')
    parser.add_argument('--infer-model-path', type=str, default='../output', help='Model path to infer from')
    parser.add_argument('--infer-file-path', type=str, help='File path to infer on')
    
    
    args = parser.parse_args()
    
    if args.train:
        # The training Process
        print('Loading data to train on')
        data_x, data_y = load_train_data()
        model = train(data_x, data_y, framework=args.framework, model_save_path=MODEL_SAVE_PATH)
    elif args.infer:
        # The inference Process
        print('Loading Model to infer on')
        
        if args.framework == 'keras':
            model = load_model(args.infer_model_path + '/keras')
        else:
            with open(args.infer_model_path + '/sklearn_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
        data_x = load_infer_data(args.infer_file_path)
        predicted_emotion = model.predict(data_x)
        predicted_emotion_index = np.argmax(predicted_emotion)+1
        print("Predicted Emotion: ", EMOTIONS_LABEL['0'+str(predicted_emotion_index)])
    
    
    