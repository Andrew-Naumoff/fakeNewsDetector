# Naive-Bayes-C.py

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


# Function to train the Naive Bayes model
def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = nb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return nb_model, accuracy, precision, recall
