import numpy as np
import pandas as pd
import re
import string
import spacy
import joblib
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

from spacy.lang.en import English
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#nltk.download('stopwords')

# Loading CSV file
df_reddit = pd.read_csv("reddit_comments.csv")

#Add the new column which gives a unique number to each of these labels
df_reddit['label_num'] = df_reddit['Label'].map({
    'Medical Doctor': 0,
    'Veterinarian': 1,
    'Other': 2
})

#check the results with top 5 rows
df_reddit.head(5)

df_reddit.Label.value_counts()

# @title Label

from matplotlib import pyplot as plt
import seaborn as sns

df_reddit.groupby('Label').size().plot(kind='barh',
                                       color=sns.palettes.mpl_palette('Dark2'))
plt.gca().spines[[
    'top',
    'right',
]].set_visible(False)

# Create our list of stopwords
stop_words = set(stopwords.words('english'))
from sklearn.feature_extraction.text import TfidfVectorizer

#spacy.download("en_core_web_lg")
nlp = spacy.load('en_core_web_lg')


def preprocess_text(text):
    # Tokenize the text
    doc = nlp(text)

    # Lemmatize the tokens and remove stop words
    cleaned_text = [token.lemma_ for token in doc if not token.is_stop]

    # Join the tokens back into a string
    cleaned_text = ' '.join(cleaned_text)

    return cleaned_text


preprocessed_texts = [
    preprocess_text(text) for text in df_reddit['Reddit Comment']
]

labels = df_reddit['label_num']
X_train, X_test, y_train, y_test = train_test_split(preprocessed_texts,
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=42)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_train_vec = X_train_vec.toarray()
X_test_vec = vectorizer.transform(X_test)
X_test_vec = X_test_vec.toarray()

#save the vectorizer
filename = 'tfidf_vectorizer.joblib'  # Replace with your desired filename
joblib.dump(vectorizer, filename)

# Train the RandomForestClassifier with class weights
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(
    X_train_vec,
    y_train,
)
# Evaluate the model
y_pred = clf.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# Save the model using joblib
filename = 'my_model1.pkl'  # Replace with your desired filename
joblib.dump(clf, filename)
"""##  Predicting New data"""

# Loading CSV file
new_data = pd.read_csv("/content/test Sheet1 .csv")

preprocessed_texts = [
    preprocess_text(text) for text in new_data['Reddit Comment']
]

new_data['label_num'] = new_data['Label'].map({
    'Medical Doctor': 0,
    'Veterinarian': 1,
    'Other': 2
})

#check the results with top 5 rows
df_reddit.head(5)
labels_new = new_data['label_num']

new_data_vec = vectorizer.fit_transform(preprocessed_texts)
new_data_vec = new_data_vec.toarray()

# Evaluate the model
#y_pred2 = clf.predict(new_data_vec)
