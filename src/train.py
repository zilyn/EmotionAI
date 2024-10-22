from ml_pipeline.model import train_model
from ml_pipeline.utils import load_train_data, balance_classes
from config import DATA_DIR, EMOTIONS_LABEL, MODEL_SAVE_PATH

# Load and preprocess data
X, y = load_train_data(DATA_DIR, EMOTIONS_LABEL)

# Balance the classes
X_balanced, y_balanced = balance_classes(X, y)

# Train the model with fine-tuned hyperparameters
model = train_model(X_balanced, y_balanced, learning_rate=0.0001, dropout_rate=0.4, batch_size=16, epochs=100)

# Save the model
model.save(MODEL_SAVE_PATH + '/keras')