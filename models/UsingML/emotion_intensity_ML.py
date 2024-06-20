import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Load datasets (Replace 'path_to_file' with actual file paths)
def load_data(file_path):
    return pd.read_csv(file_path, sep='\t', header=None, names=['id', 'tweet', 'emotion', 'degree'])

anger_train = load_data('anger_train.txt')
joy_train = load_data('joy_train.txt')
sadness_train = load_data('sadness_train.txt')
fear_train = load_data('fear_train.txt')

# Example for training a model for the anger dataset
def train_model(train_data):
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(train_data['tweet'], train_data['degree'], test_size=0.2, random_state=42)

    # Build a pipeline
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('ridge', Ridge(alpha=1.0))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model (optional)
    val_predictions = model.predict(X_val)
    mse = np.mean((val_predictions - y_val) ** 2)
    print(f'Mean Squared Error on validation set: {mse}')

    return model

anger_model = train_model(anger_train)
joy_model = train_model(joy_train)
sadness_model = train_model(sadness_train)
fear_model = train_model(fear_train)

# Function to predict and save results
def predict_and_save(model, test_data_path, output_path):
    test_data = pd.read_csv(test_data_path, sep='\t', header=None, names=['id', 'tweet', 'emotion', 'degree'])
    predictions = model.predict(test_data['tweet'])
    test_data['degree'] = predictions
    test_data.to_csv(output_path, sep='\t', header=False, index=False)

# Predict and save results for test datasets (Replace 'path_to_test' and 'path_to_output' with actual paths)
predict_and_save(anger_model, 'anger_test.txt', 'predicted_anger.txt')
predict_and_save(joy_model, 'joy_test.txt', 'predicted_joy.txt')
predict_and_save(sadness_model, 'sadness_test.txt', 'predicted_sadness.txt')
predict_and_save(fear_model, 'fear_test.txt', 'predicted_fear.txt')
