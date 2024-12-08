import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if you don't have it already
nltk.download('stopwords')

# Load the trained model and vectorizer
with open('./NaiveBayes/models/naivebayes-taglish-model.pkl', 'rb') as model_file:
    nb_model = pickle.load(model_file)

with open('./NaiveBayes/models/naivebayes-taglish-vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load English stopwords
stop_words_en = set(stopwords.words('english'))

# Load Filipino/Tagalog stopwords from a file
def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return set(stopwords)

stop_words_tl = load_stopwords('./NaiveBayes/filipino_stopwords.txt')  # Adjust the path to your stopwords file
stop_words = stop_words_en.union(stop_words_tl)

# Preprocess text: remove special characters, lowercase, remove stopwords, and handle extra spaces
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Function to predict sentiment for a given review
def predict_sentiment(review_text):
    # Preprocess the input review
    processed_review = preprocess_text(review_text)
    
    # Vectorize the processed review
    review_vec = vectorizer.transform([processed_review])
    
    # Make a prediction using the classifier
    prediction = nb_model.predict(review_vec)
    prediction_prob = nb_model.predict_proba(review_vec)
    
    return prediction[0], prediction_prob[0]

# Function to analyze sentiment in a CSV file
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

    # Analyze sentiment for each text
    sentiments = []
    confidence_negative = []
    confidence_neutral = []
    confidence_positive = []

    for text in texts:
        sentiment, sentiment_probabilities = predict_sentiment(text)
        sentiments.append(sentiment)
        confidence_negative.append(sentiment_probabilities[0])
        confidence_neutral.append(sentiment_probabilities[1])
        confidence_positive.append(sentiment_probabilities[2])

    # Add results to the dataframe
    df["Predicted Sentiment"] = sentiments
    df["Confidence Negative"] = confidence_negative
    df["Confidence Neutral"] = confidence_neutral
    df["Confidence Positive"] = confidence_positive

    # Save the updated dataframe to a new CSV file
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Results saved to {output_csv}")

# Main script
if __name__ == "__main__":
    # Specify paths and column names
    input_csv = "./NaiveBayes/naive_bayes_sentiments/taglish_sentences.csv"  # Path to the input CSV file
    output_csv = "./NaiveBayes/naive_bayes_sentiments/taglish_output.csv"  # Path to save the output CSV file
    text_column = "text"  # Column name containing text to analyze

    try:
        analyze_csv(input_csv, output_csv, text_column)
    except Exception as e:
        print(f"Error: {e}")
