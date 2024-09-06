# loadModel.py

import pickle
import os


# Function to load the model and vectorizer
def load_model(model_path='models/fake_news_classifier.pkl', vectorizer_path='models/tfidf_vectorizer.pkl'):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return model, vectorizer


# Function to predict whether the input is real or fake news
def predict(text, model, vectorizer):
    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)
    return 'Real News' if prediction == 1 else 'Fake News'
