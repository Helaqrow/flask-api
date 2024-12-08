import pandas as pd
import torch
from transformers import BartForConditionalGeneration, AutoTokenizer, pipeline
from langdetect import detect
import re

# Load the locally saved model and tokenizer from .bin and related files
def load_local_model(model_dir):
    print(f"Loading model from '{model_dir}'...")

    # Load the model from pytorch_model.bin in the directory
    model_path = f"{model_dir}/pytorch_model.bin"
    model = BartForConditionalGeneration.from_pretrained(model_dir)  # Load from directory
    model.load_state_dict(torch.load(model_path))  # Load weights
    model.eval()  # Set model to evaluation mode
    
    # Load the tokenizer from the same directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)  # Auto-detect tokenizer files
    return pipeline("summarization", model=model, tokenizer=tokenizer)

# Function to clean up warning messages and irrelevant parts
def clean_summary(summary):
    summary = re.sub(r"Your max_length is set to \d+, but your input_length is only \d+\..*", "", summary)
    summary = re.sub(r"Since this is a summarization task, where outputs shorter than the input are typically wanted, you might consider decreasing max_length manually, e.g. summarizer\('.*', max_length=\d+\)", "", summary)
    return summary.strip()

# Function to ensure the summary ends with proper punctuation
def ensure_punctuation(summary):
    if not summary.endswith(('.', '?', '!')): 
        summary += '.'
    return summary

# Function to check if a summary pertains to event evaluation
def is_event_related(summary):
    event_keywords = [
        "event", "experience", "participants", "organizers", "organizer", "performance",
        "better", "best", "amazing", "noisy", "mannerless", "improved", "disappointed"
    ]
    return any(keyword in summary.lower() for keyword in event_keywords)

# Function to summarize, clean, and filter the CSV file using batch processing
def summarize_and_filter_csv(file_path, model_dir, batch_size=8):
    # Load the summarizer pipeline with the local model
    summarizer = load_local_model(model_dir)

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Assuming the text is in a column named 'text' (adjust based on your CSV structure)
    text_column = df['text']

    # Add new columns for summaries and event-related summaries
    df['summary'] = ""
    df['Event-Related Summary'] = ""  # Default to a single space for non-event related summaries

    # Collect texts in batches
    batch_texts = []
    indices_to_process = []
    for idx, text in enumerate(text_column):
        language = detect(text)
        if language == 'tl' or language == 'en':
            batch_texts.append(text)
            indices_to_process.append(idx)

            # Process batch when it reaches the batch size or last item
            if len(batch_texts) == batch_size or idx == len(text_column) - 1:
                try:
                    # Summarize the batch of texts
                    summaries = summarizer(batch_texts, max_length=50, min_length=20, do_sample=False)

                    for i, summary in enumerate(summaries):
                        cleaned_summary = clean_summary(summary['summary_text'])
                        cleaned_summary = ensure_punctuation(cleaned_summary)
                        df.at[indices_to_process[i], 'summary'] = cleaned_summary

                        # Check if the summary is event-related
                        if is_event_related(cleaned_summary):
                            df.at[indices_to_process[i], 'Event-Related Summary'] = cleaned_summary
                        else:
                            df.at[indices_to_process[i], 'Event-Related Summary'] = "-"
                    
                    # Clear the batch for the next set of texts
                    batch_texts = []
                    indices_to_process = []

                except Exception as e:
                    for i in indices_to_process:
                        df.at[i, 'summary'] = "---ERROR: COULD NOT SUMMARIZE TEXT---"
                        df.at[i, 'Event-Related Summary'] = " "
                    print(f"Error summarizing text at index {idx}: {e}")
        else:
            print(f"Skipping text at index {idx}: Language not supported.")

    # Replace NaN in 'summary' with error message
    df['summary'] = df['summary'].fillna("---ERROR: COULD NOT SUMMARIZE TEXT---")

    df.to_csv(file_path, index=False)
    print(f"Summarization complete! Results saved in '{file_path}'.")

# Example: Call the function with the path to your CSV file
if __name__ == "__main__":
    file_path = 'final_output.csv'  # Replace with your CSV file path
    model_dir = './Summarizer/bart_model/'  # Path to the directory containing pytorch_model.bin and tokenizer files
    summarize_and_filter_csv(file_path, model_dir)
