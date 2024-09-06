# TF-IDF.py

from sklearn.feature_extraction.text import TfidfVectorizer

# Function to apply TF-IDF vectorization
def apply_tfidf(df, max_features=5000):
    tfidf = TfidfVectorizer(max_features=max_features)
    X = tfidf.fit_transform(df['cleaned_text'])
    return X, tfidf
