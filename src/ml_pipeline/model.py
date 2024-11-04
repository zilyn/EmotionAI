import configparser
import pickle
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

config = configparser.RawConfigParser()
config.read('input/config.ini')
        
HIDDEN_LAYER_SHAPE = eval(config.get('MODEL', 'hidden_layer_shape'))
BATCH_SIZE         = config.getint('MODEL', 'batch_size')
EPOCH              = config.getint('MODEL', 'epoch')
INPUT_SHAPE        = config.getint('MODEL', 'num_features')

EMOTIONS_LABEL     = eval(config.get('DATA', 'emotions'))
LEARN_EMOTIONS     = eval(config.get('DATA', 'learn_emotions'))

def get_model_keras(input_shape, num_classes):
    # Create the model
    model = Sequential()
    
    model.add(Dense(HIDDEN_LAYER_SHAPE[0], input_shape=input_shape, activation='relu'))
    for l in HIDDEN_LAYER_SHAPE[1:]:
        model.add(Dense(l, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Configure the model and start training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_model_sklearn():
    #SKlearn model defination
    model = MLPClassifier(
        alpha=0.01, batch_size=BATCH_SIZE, epsilon=1e-08,
        hidden_layer_sizes=tuple(HIDDEN_LAYER_SHAPE), learning_rate='adaptive',
        max_iter=EPOCH
    )
    return model
    
def train_keras(data_x, data_y):
    # Data
    reverse_emotions = {v:k for k,v in EMOTIONS_LABEL.items()}
    num_classes = len(LEARN_EMOTIONS)
    x_train,x_test,y_train,y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=9)
    y_train = to_categorical([int(reverse_emotions[x])-1 for x in y_train], num_classes=num_classes)
    y_test = to_categorical([int(reverse_emotions[x])-1 for x in y_test], num_classes=num_classes)

    # Configuration options
    feature_vector_length = x_train.shape[1]
    input_shape = (feature_vector_length,)
    assert x_train.shape[1]==INPUT_SHAPE
    
    model = get_model_keras(input_shape, num_classes)
    model.fit(x_train, y_train, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=0, validation_split=0.2)

    # Test the model after training
    test_results = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
    return model
    
    
def train_sklearn(data_x, data_y):
    x_train,x_test,y_train,y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=9)

    # Scikit learn model MLP
    model = get_model_sklearn()
    model.fit(x_train,y_train)
    
    # Test the model after training
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
    print("Test Result - Accuracy: {:.2f}%".format(accuracy*100))
    return model
    
    
def train(data_x, data_y, framework=None, model_save_path='./'):
    # Lets train the model with apropriate framewotk and model save path
    if framework==None or framework=='keras':
        model_save_path +='/keras' 
        model = train_keras(data_x, data_y)
        model.save(model_save_path)
    else:
        model_save_path += '/sklearn_model.pkl'
        model = train_sklearn(data_x, data_y)
        with open(model_save_path, 'wb') as f:
            pickle.dump(model, f)

    print('model saved at: ', model_save_path)
    return model

def get_model(framework):
    # get the model for trainig based on the framework specified
    if framework==None or framework=='keras':
        return get_model_keras((INPUT_SHAPE,), num_classes=len(LEARN_EMOTIONS))
    else:
        return get_model_sklearn()
