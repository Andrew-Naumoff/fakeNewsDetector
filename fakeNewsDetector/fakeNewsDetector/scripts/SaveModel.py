# SaveModel.py

import pickle

# Function to save the model and vectorizer
def save_model(model, vectorizer, model_path='models/fake_news_classifier.pkl', vectorizer_path='models/tfidf_vectorizer.pkl'):
    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    with open(vectorizer_path, 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
