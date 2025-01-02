import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load and Explore Data
def load_data(filepath):
    try:
        if filepath.endswith('.csv'):
            data = pd.read_csv(filepath)
        elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
            data = pd.read_excel(filepath)
        elif filepath.endswith('.json'):
            data = pd.read_json(filepath)
        elif filepath.endswith('.pdf'):
            # Add PDF reading functionality using PyPDF2 or pdfplumber
            import pdfplumber
            with pdfplumber.open(filepath) as pdf:
                text = ''.join(page.extract_text() for page in pdf.pages)
            data = pd.DataFrame({'text': [text], 'label': ['unknown']})  # Example format
        else:
            print("Unsupported file format.")
            return None
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Step 2: Preprocess Data
def preprocess_data(data):
    try:
        print("Available columns in the dataset:", data.columns.tolist())
        text_column = input("Enter the column name for text data: ")
        label_column = input("Enter the column name for labels: ")
        X = data[text_column]
        y = data[label_column]
        return X, y
    except KeyError as e:
        print(f"Column error: {e}")
        return None, None

# Step 3: Vectorize Text Data
def vectorize_text(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer

# Step 4: Train Model
def train_model(X_train_vec, y_train):
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model

# Step 5: Evaluate Model
def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

if __name__ == "__main__":
    # Ask user for the file path
    filepath = input("Please enter the file path for the dataset: ")

    # Load data
    data = load_data(filepath)
    if data is not None:
        
        # Preprocess data
        X, y = preprocess_data(data)
        if X is not None and y is not None:
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Vectorize text data
            X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

            # Train model
            model = train_model(X_train_vec, y_train)

            # Evaluate model
            accuracy, report = evaluate_model(model, X_test_vec, y_test)
            
            print("Accuracy:", accuracy)
            print("Classification Report:\n", report)
        else:
            print("Preprocessing failed.")
    else:
        print("Failed to load data.")
