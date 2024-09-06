# loadScript.py

from scripts.Preprocessing import load_dataset
from scripts.TF_IDF import apply_tfidf   # Import apply_tfidf
from scripts.Naive_Bayes_C import train_model
from scripts.SaveModel import save_model
from scripts.loadModel import load_model, predict

# Main function
if __name__ == "__main__":
    # Load and preprocess dataset
    df = load_dataset('data/train.csv')

    # Apply TF-IDF
    X, tfidf_vectorizer = apply_tfidf(df)

    # Get labels (adjust for your dataset)
    y = df['label']

    # Train Naive Bayes classifier
    model, accuracy, precision, recall = train_model(X, y)

    print(f"Model Accuracy: {accuracy}")
    print(f"Model Precision: {precision}")
    print(f"Model Recall: {recall}")

    # Save the trained model and vectorizer
    save_model(model, tfidf_vectorizer)

    # Load the model (for demonstration)
    loaded_model, loaded_vectorizer = load_model()

    # Make a prediction on a sample text
    sample_text = "Your news article text here."
    result = predict(sample_text, loaded_model, loaded_vectorizer)
    print(f"Prediction: {result}")

