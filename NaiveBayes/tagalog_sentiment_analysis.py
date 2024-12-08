import pickle
import pandas as pd
import numpy as np
import re

# Load the stopwords from the file
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)

stop_words = load_stopwords('./NaiveBayes/filipino_stopwords.txt')  # Load stopwords from the provided file

# Load the trained model and vectorizer
with open('./NaiveBayes/models/naivebayes-tagalog-model.pkl', 'rb') as model_file:
    nb_model = pickle.load(model_file)

with open('./NaiveBayes/models/naivebayes-tagalog-vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Preprocess text: remove special characters, make lowercase, and remove stopwords
def preprocess_text(text):
    # Remove special characters using regex
    text = re.sub(r'[^\w\s]', '', text)  # Remove anything that is not a word or space
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Calculate sentiment score function
def calculate_sentiment_score(probabilities):
    positive_prob = probabilities[0][1]  # Probability of positive class
    negative_prob = probabilities[0][0]  # Probability of negative class
    sentiment_score = positive_prob - negative_prob
    return sentiment_score

# Classify sentiment function
def classify_sentiment(sentiment_score):
    if sentiment_score > 0.1:
        return "Positive"
    elif sentiment_score < -0.1:
        return "Negative"
    else:
        return "Neutral"

# Analyze sentiment in a CSV file
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

    # Preprocess and vectorize Tagalog sentences
    preprocessed_texts = [preprocess_text(text) for text in texts]
    vectorized_texts = vectorizer.transform(preprocessed_texts)

    # Predict sentiment labels and probabilities for each text
    sentiments = []
    sentiment_scores = []
    for vectorized_text in vectorized_texts:
        # Ensure the input is in the correct 2D shape (even for single sample)
        vectorized_text = vectorized_text.reshape(1, -1)  # Reshape to 2D (1 sample, n features)
        probabilities = nb_model.predict_proba(vectorized_text)
        sentiment_score = calculate_sentiment_score(probabilities)
        sentiment_label = classify_sentiment(sentiment_score)
        sentiments.append(sentiment_label)
        sentiment_scores.append(sentiment_score)

    # Add results to the dataframe
    df["Predicted Sentiment"] = sentiments
    df["Sentiment Score"] = sentiment_scores

    # Save the updated dataframe to a new CSV file
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Results saved to {output_csv}")

# Main script
if __name__ == "__main__":
    # Specify paths and column names
    input_csv = "./NaiveBayes/naive_bayes_sentiments/tagalog_sentences.csv"  # Path to the input CSV file
    output_csv = "./NaiveBayes/naive_bayes_sentiments/tagalog_output.csv"  # Path to save the output CSV file
    text_column = "text"  # Column name containing text to analyze

    try:
        analyze_csv(input_csv, output_csv, text_column)
    except Exception as e:
        print(f"Error: {e}")
