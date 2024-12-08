import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn.functional as F

# Load the trained model and tokenizer
model_path = './BERT/Models/taglish'  # Path to your new BERT model
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define the sentiment mapping
sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}

# Function to analyze sentiment for a list of texts
def analyze_sentiment(texts):
    results = []
    for text in texts:
        # Tokenize the input text
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )

        # Move tokens to the same device as the model
        tokens = {key: val.to(device) for key, val in tokens.items()}

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
            logits = outputs.logits

            # Apply softmax to get probabilities
            probabilities = F.softmax(logits, dim=-1)
            predicted_label = torch.argmax(probabilities, dim=-1).item()
            confidence_score = probabilities[0, predicted_label].item()

        # Map the label to its sentiment
        sentiment = sentiment_map[predicted_label]
        results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence_score
        })
    return results

# Function to process a CSV file for sentiment analysis
def analyze_csv(input_csv, output_csv, text_column):
    """Analyze sentiment for text data in a CSV file."""
    # Load the CSV file
    try:
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
    print("Analyzing sentiments...")
    results = analyze_sentiment(texts)

    # Add results to the dataframe
    df["Predicted Sentiment"] = [result["sentiment"] for result in results]
    df["Confidence"] = [result["confidence"] for result in results]

    # Save the updated dataframe to a new CSV file
    df.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"Results saved to {output_csv}")

# Main script
if __name__ == "__main__":
    # Specify paths and column names
    input_csv = "./BERT/BERT_sentiments/taglish_sentences.csv"  # Path to the input CSV file
    output_csv = "./BERT/BERT_sentiments/taglish_output.csv"  # Path to save the output CSV file
    text_column = "text"  # Column name containing text to analyze

    # Example usage for CSV processing
    try:
        analyze_csv(input_csv, output_csv, text_column)
    except Exception as e:
        print(f"Error: {e}")
