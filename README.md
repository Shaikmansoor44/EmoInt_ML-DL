# EmoInt_ML-DL
Develop methods to identify the intensity of emotions in text
# Predicting Emotion Intensity in Tweets

## Introduction
This project aims to predict the intensity of various emotions in tweets using two distinct approaches: a purely statistical model combining TF-IDF vectorization and Ridge Regression, and a deep learning model based on BERT (Bidirectional Encoder Representations from Transformers). 

## Objectives
1. Develop a purely statistical model using TF-IDF vectorization and Ridge Regression to predict the intensity of emotions in tweets.
2. Implement a deep learning model using BERT to achieve the same goal.

## Dataset
The datasets for this project are provided in tab-separated files with four columns: `id`, `tweet`, `emotion`, and `degree`. The `degree` column represents the intensity of the emotion in the tweet. There are separate training and testing datasets for each emotion: anger, joy, sadness, and fear.

## Methodologies

### 1. Purely Statistical Model

#### TF-IDF Vectorization and Ridge Regression
- **TF-IDF Vectorization**: Transforms text data into numerical features by measuring the importance of words based on their frequency and uniqueness across documents.
  - **Term Frequency (TF)**: Measures how frequently a term occurs in a document.
  - **Inverse Document Frequency (IDF)**: Measures how important a term is by weighing down frequent terms while scaling up rare ones.
  - **TF-IDF**: The product of TF and IDF scores for a term.

- **Ridge Regression**: A linear model that mitigates multicollinearity by introducing a regularization term, making it effective for high-dimensional data like TF-IDF vectors.

#### Process
1. **Data Loading**: Data is loaded from the provided files into Pandas DataFrames.
2. **Data Splitting**: The data is split into training (80%) and validation (20%) sets.
3. **Pipeline Construction**: A pipeline combining TF-IDF vectorization and Ridge Regression is built.
4. **Model Training**: The model is trained on the training set.
5. **Evaluation**: The model's performance is evaluated using Mean Squared Error (MSE) on the validation set.
6. **Prediction**: Predictions are made on the test datasets, and results are saved to new files.

#### Results
The MSE for each emotion's validation set is as follows:
- **Anger Model**: MSE = 0.021378882821692174
- **Joy Model**: MSE = 0.030532023503292736
- **Sadness Model**: MSE = 0.02552532835528938
- **Fear Model**: MSE = 0.024241308988040174

### 2. Deep Learning Model

#### BERT Model
- **BERT**: A powerful NLP model that processes text bidirectionally, considering the context from both the left and right of each word. It is pre-trained on a large corpus of text and fine-tuned for specific tasks.

#### Process
1. **Data Loading and Preprocessing**: Datasets are loaded and preprocessed, including handling 'NONE' values in the test data scores.
2. **Tokenization**: Tweets are tokenized using the BERT tokenizer, ensuring uniformity with padding and truncation to a maximum length of 128 tokens.
3. **Model Integration**: The pre-trained BERT model is used as the base, with a regression head added on top for predicting emotion intensity.
4. **Training**: The model is trained separately for each emotion category, using the tokenized training data.
5. **Prediction**: The trained model predicts scores for the test set, and the results are integrated back into the test data for each emotion category.

### Conclusion
- **TF-IDF and Ridge Regression**: Offers a computationally efficient and interpretable approach for predicting emotion intensity, suitable for smaller datasets or resource-constrained scenarios. However, it may not capture intricate semantic relationships and contextual nuances as effectively as deep learning models.
- **BERT**: Utilizes advanced language understanding capabilities to provide highly accurate predictions, demonstrating the potential of transformer-based models in understanding and analyzing human emotions in text.

### Future Work
Future improvements could include:
- Experimenting with different regression models.
- Tuning hyperparameters.
- Incorporating more sophisticated text preprocessing techniques.
- Exploring hybrid approaches that combine the strengths of traditional methods with deep learning techniques to enhance performance in capturing nuanced emotional expressions in text data.

## Files
- `anger_train.txt`, `anger_test.txt`: Training and testing datasets for anger.
- `joy_train.txt`, `joy_test.txt`: Training and testing datasets for joy.
- `sadness_train.txt`, `sadness_test.txt`: Training and testing datasets for sadness.
- `fear_train.txt`, `fear_test.txt`: Training and testing datasets for fear.
- `predicted_anger.txt`, `predicted_joy.txt`, `predicted_sadness.txt`, `predicted_fear.txt`: Files containing the predicted emotion intensities for the test datasets.
