# Preprocessing.py

import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

# Ensure the stopwords are downloaded
nltk.download('stopwords')

# Preprocess function to clean the text
def preprocess_text(text):
    # Ensure the input is a string
    if not isinstance(text, str):
        return ''
    stop_words = set(stopwords.words('english'))
    # Remove punctuation and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Convert text to lowercase and remove stop words
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])

    return text

# Load the dataset
def load_dataset(filepath):
    df = pd.read_csv(filepath)
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    return df
