# Import necessary libraries
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load the trained model and tokenizer from the output directory
tokenizer = AutoTokenizer.from_pretrained("./BERT/Models/tagalog")
trained_model = AutoModelForSequenceClassification.from_pretrained("./BERT/Models/tagalog")

# Define a sentiment analysis pipeline using the trained model
sentiment_pipeline = pipeline("sentiment-analysis", model=trained_model, tokenizer=tokenizer)

# Define a mapping dictionary to map encoded labels to human-readable forms
label_mapping = {
    "LABEL_0": "negative",
    "LABEL_1": "positive",
    # Add more mappings as needed
}

# Function to analyze sentiment for a single text
def analyze_sentiment(text, threshold=0.65):
    max_length = tokenizer.model_max_length

    # Split the text into chunks of maximum token length
    chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]

    # Initialize lists to store sentiment results
    sentiment_labels = []
    sentiment_scores = []

    # Analyze sentiment for each chunk
    for chunk in chunks:
        sentiment_result = sentiment_pipeline(chunk)
        sentiment_labels.append(sentiment_result[0]["label"])
        sentiment_scores.append(sentiment_result[0]["score"])

    # Get the sentiment label and map it to human-readable form
    encoded_label = max(sentiment_labels, key=sentiment_labels.count)
    sentiment_label = label_mapping.get(encoded_label, "unknown")

    # Get the average sentiment score
    positive_score = sum(sentiment_scores) / len(sentiment_scores)

    # Classify sentiment based on the positive score using threshold logic
    if 0.35 <= positive_score <= 0.65:
        sentiment_label = "neutral"
    elif positive_score > 0.65:
        sentiment_label = "positive"
    else:
        sentiment_label = "negative"

    return sentiment_label, positive_score

# Function to process a CSV file for sentiment analysis
def process_csv(input_csv, output_csv, text_column):
    """Process a CSV file, predict sentiments, and save results."""
    try:
        # Read the input CSV file
        df = pd.read_csv(input_csv, encoding="utf-8")  # Adjust encoding if needed
    except UnicodeDecodeError as e:
        print(f"Encoding Error: {e}")
        return

    # Ensure the text column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the input CSV file.")

    # Clean the text column: replace NaN with empty strings
    df[text_column] = df[text_column].fillna("").astype(str)

    # Analyze sentiment for each row
    sentiments = []
    scores = []
    for text in df[text_column]:
        try:
            sentiment, score = analyze_sentiment(text)
            sentiments.append(sentiment)
            scores.append(score)
        except Exception as e:
            print(f"Error processing text: {text} -> {e}")
            sentiments.append("error")
            scores.append(None)

    # Add results to the dataframe
    df["Predicted Sentiment"] = sentiments
    df["Confidence"] = scores

    # Save the updated dataframe to a new CSV file
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Results saved to {output_csv}")

# Main script
if __name__ == "__main__":
    # Specify file paths
    input_csv = "./BERT/BERT_sentiments/tagalog_sentences.csv"  # Replace with your input CSV file
    output_csv = "./BERT/BERT_sentiments/tagalog_output.csv"  # Replace with your output file
    text_column = "text"  # Replace with the column name containing text to analyze

    try:
        process_csv(input_csv, output_csv, text_column)
    except Exception as e:
        print(f"Error: {e}")
