import pandas as pd

# Define paths to datasets
datasets = {
    'joy': {
        'train': 'joy_train.txt',
        'test': 'joy_test.txt'
    },
    'sadness': {
        'train': 'sadness_train.txt',
        'test': 'sadness_test.txt'
    },
    'fear': {
        'train': 'fear_train.txt',
        'test': 'fear_test.txt'
    },
    'anger': {
        'train': 'anger_train.txt',
        'test': 'anger_test.txt'
    }
}

def load_data(file_path):
    return pd.read_csv(file_path, delimiter='\t', header=None, names=['id', 'tweet', 'emotion', 'score'])

# Load all datasets
train_data = {emotion: load_data(paths['train']) for emotion, paths in datasets.items()}
test_data = {emotion: load_data(paths['test']) for emotion, paths in datasets.items()}

# Replace 'NONE' in test_data scores with NaN
for emotion, data in test_data.items():
    data['score'].replace('NONE', pd.NA, inplace=True)



import tensorflow as tf
from transformers import BertTokenizer

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set max length for tokenization
MAX_LENGTH = 128

# Tokenize tweets
def tokenize_tweets(tweets):
    return tokenizer(
        tweets.tolist(),
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

train_encodings = {emotion: tokenize_tweets(data['tweet']) for emotion, data in train_data.items()}
test_encodings = {emotion: tokenize_tweets(data['tweet']) for emotion, data in test_data.items()}



from transformers import TFBertModel

# Load pre-trained BERT model
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Define a regression model on top of BERT
def build_model():
    input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int32, name='attention_mask')

    bert_output = bert_model(input_ids, attention_mask=attention_mask)[1]  # Use pooled output
    output = tf.keras.layers.Dense(1, activation='linear')(bert_output)  # Regression output

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5), loss='mean_squared_error')
    return model



# Train and predict for each emotion
results = {}

for emotion in datasets.keys():
    print(f'Training and predicting for {emotion}...')

    model = build_model()

    # Prepare inputs
    train_inputs = {
        'input_ids': train_encodings[emotion]['input_ids'],
        'attention_mask': train_encodings[emotion]['attention_mask']
    }

    # Train the model
    model.fit(
        train_inputs,
        train_data[emotion]['score'].astype(float).values,
        validation_split=0.1,
        epochs=3,
        batch_size=16
    )

    # Prepare test inputs
    test_inputs = {
        'input_ids': test_encodings[emotion]['input_ids'],
        'attention_mask': test_encodings[emotion]['attention_mask']
    }

    # Predict scores for the test set
    predictions = model.predict(test_inputs)

    # Add predictions to test data
    test_data[emotion]['score'] = predictions

    # Save the results
    results[emotion] = test_data[emotion]
    output_file = f'predicted_{emotion}_emotion.txt'
    test_data[emotion].to_csv(output_file, sep='\t', header=False, index=False)

print('predictions completed.')
