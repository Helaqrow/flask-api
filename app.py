from flask import Flask, render_template, request, jsonify
import subprocess
import os
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Define paths for input and output files
base_csv = "base.csv"  # The input CSV with mixed English, Tagalog, and Taglish texts
language_detected_csv = "./language_detected.csv"  # Temporary file to store language-detected results
final_output_csv = "final_output.csv"

# Path to your existing scripts
language_detector_script = "./language_identifier/language_detectorTest.py"
english_sentiment_script = "./BERT/engBERTCSV.py"
filipino_sentiment_script = "./BERT/filipinoBERTCSV.py"
taglish_sentiment_script = "./BERT/taglishBERTCSV.py"
predictive_analysis_script = "./Predictive_Analysis/predictive_prophet.py"
summarizer_script = "./Summarizer/summarize_bart.py"  # Path to the summarizer script

# Function to delete intermediate files from a previous run
def cleanup_files():
    files_to_delete = [
        "./BERT/BERT_sentiments/english_output.csv",
        "./BERT/BERT_sentiments/tagalog_output.csv",
        "./BERT/BERT_sentiments/taglish_output.csv",
        final_output_csv,
        './static/sentiment_trend_forecast.png'  # This assumes the forecast image is saved with this name
    ]
    
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

# Function to clear or delete the base.csv file if it exists
def clear_base_csv():
    if os.path.exists(base_csv):
        os.remove(base_csv)  # Delete the file if it exists
        print("Existing 'base.csv' has been deleted.")

# Function to search and collect matching rows from the CSV file
def search_and_collect(input_text):
    csv_file = 'post_data.csv'
    base_csv = 'base.csv'  # File to save the filtered results
    try:
        df = pd.read_csv(csv_file, encoding='ISO-8859-1')  # Try using ISO-8859-1 encoding
    except UnicodeDecodeError as e:
        print(f"Error reading the CSV file: {e}")
        return None

    # Ensure the 'text' and 'date' columns exist in the CSV
    if 'text' not in df.columns or 'date' not in df.columns:
        print("Error: 'text' or 'date' column not found in the CSV file.")
        return None

    # Filter rows where the 'text' column matches the user input (case-insensitive)
    matching_rows = df[df['text'].str.contains(input_text, case=False, na=False)]

    if matching_rows.empty:
        print(f"No matches found for the input: {input_text}")
        return None  # Return None to indicate no match found
    else:
        selected_data = matching_rows[['text', 'date']]
        selected_data['text'] = selected_data['text'].str.replace(r'\\n', ' ', regex=True)
        selected_data['sentiment'] = ''
        selected_data.to_csv(base_csv, index=False)
        print(f"Collected data has been saved to '{base_csv}'.")
        return True  # Indicate successful match

# Function to run a subprocess (Python script) for each stage of the pipeline
def run_subprocess(script_name, args):
    try:
        subprocess.run(["python", script_name] + args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        raise

# Function to merge results from different sentiment analysis outputs
def merge_results():
    try:
        df_english = pd.read_csv("./BERT/BERT_sentiments/english_output.csv", encoding="utf-8")
        df_tagalog = pd.read_csv("./BERT/BERT_sentiments/tagalog_output.csv", encoding="utf-8")
        df_taglish = pd.read_csv("./BERT/BERT_sentiments/taglish_output.csv", encoding="utf-8")
        final_df = pd.concat([df_english, df_tagalog, df_taglish], ignore_index=True)
        final_df.to_csv(final_output_csv, index=False, encoding="utf-8")
        print(f"Results successfully merged into {final_output_csv}")
    except Exception as e:
        print(f"Error in merging results: {e}")

# Function to rename columns in the final output CSV
def rename_columns_for_predictive():
    try:
        df = pd.read_csv(final_output_csv, encoding="utf-8")
        if 'Predicted Sentiment' in df.columns:
            df['sentiment'] = df['Predicted Sentiment']
            df.drop(columns=['Predicted Sentiment'], inplace=True)
        else:
            print("Error: 'Predicted Sentiment' column not found in the CSV.")
            raise ValueError("'Predicted Sentiment' column missing in the final output CSV.")
        if 'date' not in df.columns:
            print("Error: 'date' column not found in the CSV.")
            raise ValueError("'date' column missing in the final output CSV.")
        df.to_csv(final_output_csv, index=False, encoding="utf-8")
        print(f"'sentiment' column updated successfully in {final_output_csv}")
    except Exception as e:
        print(f"Error in renaming columns: {e}")

# Function to run the summarizer script
def summarize_output(input_csv):
    try:
        print("Running summarizer on merged output...")
        run_subprocess(summarizer_script, [input_csv])  # Call the summarizer script
        print(f"Summarized output saved to {input_csv}")
    except Exception as e:
        print(f"Error in summarizing output: {e}")

@app.route('/')
def home():
    # Cleanup files when the app loads
    cleanup_files()
    
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    input_text = request.form['search_text']
    
    # Check if the input text is less than 3 characters
    if len(input_text) < 3:
        error_message = "Error: The input text is too short. Please enter at least 3 characters."
        return render_template('results.html', error_message=error_message)
    
    # Cleanup files from the previous run
    cleanup_files()
    
    clear_base_csv()
    match_found = search_and_collect(input_text)
    
    # Check if a match was found, otherwise return error message
    if not match_found:
        error_message = f"No matches found for '{input_text}' in the CSV file."
        return render_template('results.html', error_message=error_message)
    
    try:
        run_subprocess(language_detector_script, [base_csv, language_detected_csv])
        run_subprocess(english_sentiment_script, ["./BERT/BERT_sentiments/english_sentences.csv", "./BERT/BERT_sentiments/english_output.csv", "text"])
        run_subprocess(filipino_sentiment_script, ["./BERT/BERT_sentiments/tagalog_sentences.csv", "./BERT/BERT_sentiments/tagalog_output.csv", "text"])
        run_subprocess(taglish_sentiment_script, ["./BERT/BERT_sentiments/taglish_sentences.csv", "./BERT/BERT_sentiments/taglish_output.csv", "text"])
        merge_results()
        summarize_output(final_output_csv)
        rename_columns_for_predictive()
        run_subprocess(predictive_analysis_script, [final_output_csv])
    except Exception as e:
        error_message = f"An error occurred while processing your request: {e}"
        return render_template('results.html', error_message=error_message)

    plot_file = 'sentiment_trend_forecast.png'
    final_results = pd.read_csv(final_output_csv)
    return render_template('results.html', 
                           plot_file=plot_file, 
                           tables=[final_results.to_html(classes='data', header="true")])


if __name__ == '__main__':
    # Run the app without debug mode, specifying host and port
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
