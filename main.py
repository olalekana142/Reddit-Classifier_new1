import numpy as np
import pandas as pd
from joblib import load
import re
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import streamlit as st
from sklearn.ensemble import RandomForestClassifier

model = load('my_model.pkl')
# In your new environment, load the saved vectorizer
loaded_vectorizer = load('tfidf_vectorizer.joblib')

# Preprocess the data
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
    Clean and pre-process text data.

    Args:
        text (str): The text to be cVeterinarianleaned.

    Returns:
        str: The cleaned text.
    """

    text = text.lower()  # Lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ',
                                   text)  # Replace special symbols with spaces
    text = BAD_SYMBOLS_RE.sub(
        '', text)  # Remove non-alphanumeric characters (except #, +, _)
    text = text.replace('x',
                        '')  # Remove any remaining 'x' characters (optional)

    return text


top_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer

nlp = spacy.load('en_core_web_sm')


def preprocess_text(text):
    # Tokenize the text
    doc = nlp(text)

    # Lemmatize the tokens and remove stop words
    cleaned_text = [token.lemma_ for token in doc if not token.is_stop]

    # Join the tokens back into a string
    cleaned_text = ' '.join(cleaned_text)

    return cleaned_text


# Define the mapping dictionary
label_mapping = {0: 'Medical Doctor', 1: 'Veterinarian', 2: 'Other'}

vectorizer = TfidfVectorizer()

st.title("Reddit Comment Classifier For Veterinarian and Medical Doctor")

# Display file uploader
file = st.file_uploader("Upload CSV file", type="csv")

# Check if a file was uploaded
if file is not None:
    # Read the CSV file
    df = pd.read_csv(file)

prediction_button = st.button("Predict")

if prediction_button:
    preprocessed_texts = [
        preprocess_text(text) for text in df['Reddit Comment']
    ]
    # Use the loaded vectorizer to transform new documents
    new_features = loaded_vectorizer.transform(preprocessed_texts)

    # Make a prediction
    prediction = model.predict(new_features)
    # Format the prediction output

    # Map the numeric labels to string labels

    # Create a DataFrame from the predictions
    label_num = pd.DataFrame(prediction, columns=['prediction'])

    df['prediction_label'] = label_num['prediction'].map(label_mapping)

    # Display the processed DataFrame
    st.dataframe(df)

# st.write("Prediction:", prediction)
