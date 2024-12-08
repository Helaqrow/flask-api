import subprocess
import os
import pandas as pd

# Define file paths
base_csv = "base.csv"  # The input CSV with mixed English, Tagalog, and Taglish texts
language_detected_csv = "./language_detected.csv"  # Temporary file to store language-detected results

# Language-specific CSV files from language detection output
tagalog_csv = "./BERT/BERT_sentiments/tagalog_sentences.csv"
english_csv = "./BERT/BERT_sentiments/english_sentences.csv"
taglish_csv = "./BERT/BERT_sentiments/taglish_sentences.csv"

# Language-specific BERT output paths
english_bert_output = "./BERT/BERT_sentiments/english_output.csv"
tagalog_bert_output = "./BERT/BERT_sentiments/tagalog_output.csv"
taglish_bert_output = "./BERT/BERT_sentiments/taglish_output.csv"

# Final merged output path
final_output_csv = "final_output.csv"

# Script paths
language_detector_script = "./language_identifier/language_detectorTest.py"
english_sentiment_script = "./BERT/engBERTCSV.py"
filipino_sentiment_script = "./BERT/filipinoBERTCSV.py"
taglish_sentiment_script = "./BERT/taglishBERTCSV.py"
summarizer_script = "./Summarizer/summarize_bart.py"  # Path to the summarizer script
predictive_analysis_script = "./Predictive_Analysis/predictive_prophet.py"

# Columns
language_column = "Detected Language"
text_column = "text"

def clean_csv_files(*file_paths):
    """Initialize the given CSV files to be empty."""
    try:
        for file_path in file_paths:
            with open(file_path, 'w', encoding='utf-8') as f:
                if "sentences" in file_path or "output" in file_path:
                    f.write("text\n")  # For files expected to contain text data
                elif "final_output" in file_path:
                    f.write("date,sentiment\n")  # For final output CSV
                else:
                    f.write("")  # Leave other files blank
        print(f"Cleaned files: {', '.join(file_paths)}")
    except Exception as e:
        print(f"Error cleaning files: {e}")

def run_subprocess(script_name, args):
    """Run a Python script as a subprocess with arguments."""
    try:
        subprocess.run(["python", script_name] + args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        raise

def merge_results(english_bert_output, tagalog_bert_output, taglish_bert_output, final_output_csv):
    """Merge the results of sentiment analysis into a final output CSV."""
    try:
        # Load the CSVs for English, Tagalog, and Taglish output
        df_english = pd.read_csv(english_bert_output, encoding="utf-8")
        df_tagalog = pd.read_csv(tagalog_bert_output, encoding="utf-8")
        df_taglish = pd.read_csv(taglish_bert_output, encoding="utf-8")
        
        # Concatenate the results from all languages
        final_df = pd.concat([df_english, df_tagalog, df_taglish], ignore_index=True)
        
        # Save the merged results to the final output CSV
        final_df.to_csv(final_output_csv, index=False, encoding="utf-8")
        print(f"Results successfully merged into {final_output_csv}")
    
    except Exception as e:
        print(f"Error in merging results: {e}")

def summarize_output(input_csv, output_csv):
    """Summarize the merged output and save it back to the same file."""
    try:
        print("Running summarizer on merged output...")
        run_subprocess(summarizer_script, [input_csv])  # Call the summarizer script
        print(f"Summarized output saved to {output_csv}")
    except Exception as e:
        print(f"Error in summarizing output: {e}")

def rename_columns_for_predictive(final_output_csv):
    """Ensure 'sentiment' column exists with correct values and remove 'Predicted Sentiment'."""
    try:
        # Load the final output CSV
        df = pd.read_csv(final_output_csv, encoding="utf-8")
        
        # Check if 'Predicted Sentiment' column exists
        if 'Predicted Sentiment' in df.columns:
            # Move contents of 'Predicted Sentiment' to 'sentiment'
            df['sentiment'] = df['Predicted Sentiment']
            # Drop the 'Predicted Sentiment' column
            df.drop(columns=['Predicted Sentiment'], inplace=True)
        else:
            print("Error: 'Predicted Sentiment' column not found in the CSV.")
            raise ValueError("'Predicted Sentiment' column missing in the final output CSV.")
        
        # Ensure 'date' column exists and is correctly named
        if 'date' not in df.columns:
            print("Error: 'date' column not found in the CSV.")
            raise ValueError("'date' column missing in the final output CSV.")
        
        # Save the updated DataFrame back to the same file
        df.to_csv(final_output_csv, index=False, encoding="utf-8")
        print(f"'sentiment' column updated successfully in {final_output_csv}")
    
    except Exception as e:
        print(f"Error in renaming columns: {e}")

# Main script
if __name__ == "__main__":
    # Step 1: Clean all CSV files
    print("Cleaning CSV files...")
    clean_csv_files(language_detected_csv, tagalog_csv, english_csv, taglish_csv,
                    english_bert_output, tagalog_bert_output, taglish_bert_output, final_output_csv)

    # Step 2: Run the language detector
    print("Running language detector...")
    run_subprocess(language_detector_script, [base_csv, language_detected_csv])

    # Step 3: Run sentiment analysis for each language
    print("Running English sentiment analysis...")
    run_subprocess(english_sentiment_script, [english_csv, english_bert_output, text_column])

    print("Running Tagalog sentiment analysis...")
    run_subprocess(filipino_sentiment_script, [tagalog_csv, tagalog_bert_output, text_column])

    print("Running Taglish sentiment analysis...")
    run_subprocess(taglish_sentiment_script, [taglish_csv, taglish_bert_output, text_column])

    # Step 4: Merge results into final output
    print("Merging results into final output...")
    merge_results(english_bert_output, tagalog_bert_output, taglish_bert_output, final_output_csv)

    # Step 5: Summarize the merged output
    print("Summarizing merged output...")
    summarize_output(final_output_csv, final_output_csv)

    # Step 6: Run predictive analysis
    print("Running predictive analysis...")
    run_subprocess(predictive_analysis_script, [final_output_csv])

    print("Pipeline complete!")
