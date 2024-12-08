import pandas as pd
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Load the new trained model and vectorizer
with open('./NaiveBayes/models/naivebayes-english-model.pkl', 'rb') as model_file:
    nb_model = pickle.load(model_file)

with open('./NaiveBayes/models/naivebayes-english-vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocess text function (same as before)
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

# Classify sentiment function (same as before)
def classify_sentiment(prediction):
    if prediction == 1:
        return "Positive"
    elif prediction == 0:
        return "Neutral"
    else:
        return "Negative"

# Analyze sentiment for text data in a CSV file
def analyze_csv(input_csv, output_csv, text_column):
    """Analyze sentiment for text data in a CSV file."""
    try:
        # Load the CSV file
        df = pd.read_csv(input_csv, encoding="utf-8")
    except UnicodeDecodeError as e:
        print(f"Encoding error: {e}")
        return

    # Ensure the text column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the input CSV file.")

    # Fill missing text with empty strings and convert to list
    texts = df[text_column].fillna("").astype(str).tolist()

    # Preprocess and vectorize the text data
    preprocessed_texts = [preprocess_text(text) for text in texts]
    vectorized_texts = vectorizer.transform(preprocessed_texts)

    # Predict sentiment labels for the texts
    predictions = nb_model.predict(vectorized_texts)
    prediction_probs = nb_model.predict_proba(vectorized_texts)

    # Calculate the sentiment score (confidence)
    sentiments = []
    sentiment_scores = []
    for i, prob in enumerate(prediction_probs):
        sentiment_label = classify_sentiment(predictions[i])
        confidence = np.max(prob) * 100  # Convert to percentage
        sentiments.append(sentiment_label)
        sentiment_scores.append(confidence)

    # Add results to the dataframe
    df["Predicted Sentiment"] = sentiments
    df["Sentiment Confidence"] = sentiment_scores

    # Save the updated dataframe to a new CSV file
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Results saved to {output_csv}")

# Main script
if __name__ == "__main__":
    # Specify paths and column names
    input_csv = "./NaiveBayes/naive_bayes_sentiments/english_sentences.csv"  # Path to the input CSV file
    output_csv = "./NaiveBayes/naive_bayes_sentiments/english_output.csv"  # Path to save the output CSV file
    text_column = "text"  # Column name containing text to analyze

    try:
        analyze_csv(input_csv, output_csv, text_column)
    except Exception as e:
        print(f"Error: {e}")
