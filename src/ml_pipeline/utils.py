import os
import glob
import configparser
import tqdm

import librosa
import soundfile
import numpy as np

config = configparser.RawConfigParser()
config.read('../input/config.ini')

DATA_DIR      = config.get('DATA','data_dir')
FILE_GLOB     = config.get('DATA','file_glob')
EMOTIONS_LABEL= eval(config.get('DATA','emotions'))
LEARN_EMOTIONS = eval(config.get('DATA','learn_emotions'))
LEARN_LABELS = [EMOTIONS_LABEL['0'+str(x)] for x in LEARN_EMOTIONS]

def load_train_data():
    '''
    Function to Load the Train data.
    The function doesn't take any parameter but relies on the GLOBAL variables from the config
    file to get the directory and file name structures.
    
    return: List of numpy arrays containing both X and Y for training. [[X....], [Y....]]
    '''
    x,y=[],[]
    # Loop over all the files that matches the directory and glob pattern
    # DIR + Audio_*_Actors_01-24/Actor_*/*.wav
    for file in tqdm.tqdm(glob.glob(DATA_DIR + FILE_GLOB)[:20]):
        try:
            file_name = os.path.basename(file)
            emotion = EMOTIONS_LABEL[file_name.split("-")[2]]
            if emotion not in LEARN_LABELS:
                continue
            feature = extract_feature(file)
            x.append(feature)
            y.append(emotion)
        except Exception as e:
            # Just skip printing the error, if any file can't be read, due to file corruption or any other case.
            print(e, file)
    
    # Print the final shape of training data X
    print('shape of loaded data: ', np.array(x).shape)
    return [np.array(x), y]

def extract_feature(file_name, mfcc=True, chroma=True, mel=True, zero_crossing=True):
    '''
    Function to extract features from a single sound file. It can derive 4 features, and can be extended to extract
    more features in the same way.
    
    Parameters
    file_name (str): File name to extract the features for.
    mfcc (bool): If the MFCC (Mel-Frequency Cepstral Coefficients) feature needs 
                    to be calculated or not. By default it's True.
    Similarly chroma, mel and zero_crossing features, all are turned on by default.
    
    return (list): [all the feature values concatenated to form a single list.]
    '''
    
    with soundfile.SoundFile(file_name) as sound_file:
        # Open the soundfile
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        # Get the sample rate of the sound
        result = np.array([])
        if mfcc:
            # Calculate Mel-Frequency Cepstral Coefficients
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            # Calculate chroma stft (chroma spectrogram)
            stft = np.abs(librosa.stft(X))
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate, n_chroma=24).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            # Calculate mel spectrogram
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)  # Added y= parameter
            result = np.hstack((result, mel))
        if zero_crossing:
            # Calculate the number of zero crossings
            zc = sum(librosa.zero_crossings(X, pad=False))
            result = np.hstack((result, zc))
    return result

def load_infer_data(filepath):
    ''' 
    Function to load the inference data from filepath and extract the features the same way we did for training.
    '''
    file_name = os.path.basename(filepath)
    feature = np.array(extract_feature(filepath))
    print(feature.shape)
    # Printing the feature shape and reshaping it to a single row vector.
    return feature.reshape(1, -1)
