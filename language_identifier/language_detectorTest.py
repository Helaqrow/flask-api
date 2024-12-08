import os  # For creating directories
import joblib
import pandas as pd
import chardet  # For detecting file encoding
import re  # For sentence splitting

# Step 1: Load the saved model and vectorizer
loaded_model = joblib.load("./language_identifier/model/language_detector_model.pkl")
loaded_vectorizer = joblib.load("./language_identifier/model/vectorizer.pkl")

# Load the lists of English and Tagalog words
with open("./language_identifier/languages/en.txt", "r", encoding="utf-8") as f:
    english_sentences = f.readlines()
english_words = set(word.lower() for sentence in english_sentences for word in sentence.split())

with open("./language_identifier/languages/tl.txt", "r", encoding="utf-8") as f:
    tagalog_sentences = f.readlines()
tagalog_words = set(word.lower() for sentence in tagalog_sentences for word in sentence.split())

# Step 2: Function to detect language
def detect_language(sentence, vectorizer, model):
    """Detect the language of a given sentence."""
    if not sentence.strip():
        return "unknown"  # Handle empty or noisy sentences

    sentence_vec = vectorizer.transform([sentence])
    lang = model.predict(sentence_vec)[0]

    # Check for Taglish (mix of English and Tagalog)
    words = sentence.split()
    english_count = sum(1 for word in words if word.lower() in english_words)
    tagalog_count = sum(1 for word in words if word.lower() in tagalog_words)

    # Detect Taglish if both English and Tagalog words meet the threshold
    if english_count >= 0.5 * (english_count + tagalog_count) and tagalog_count >= 0.4 * (english_count + tagalog_count):
        return "Taglish"

    return lang

# Step 3: Function to split text into sentences
def split_into_sentences(text):
    """Split text into sentences based on common punctuation marks."""
    # Basic sentence splitting using period, exclamation mark, or question mark.
    sentences = re.split(r'[.!?]\s+', text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]

# Step 4: Process CSV and write results to separate files
def process_csv(input_csv, tagalog_csv, english_csv, taglish_csv, text_column, 
                tagalognaivebayes_csv, englishnaivebayes_csv, taglishnaivebayes_csv):
    """Process the CSV file and separate rows based on detected language."""
    # Check file encoding using chardet
    try:
        with open(input_csv, 'rb') as f:
            result = chardet.detect(f.read())  # Detect the file encoding
        encoding = result['encoding']
        print(f"Detected file encoding: {encoding}")
    except Exception as e:
        print(f"Error detecting encoding: {e}")
        encoding = "utf-8"  # Default to utf-8 if detection fails

    # Load the CSV file with the detected encoding
    try:
        df = pd.read_csv(input_csv, encoding=encoding)
    except UnicodeDecodeError as e:
        print(f"Encoding error: {e}")
        print("Attempting with ISO-8859-1 encoding...")
        # If UTF-8 fails, attempt with a different encoding (ISO-8859-1)
        try:
            df = pd.read_csv(input_csv, encoding="ISO-8859-1")
        except Exception as ex:
            print(f"Failed with ISO-8859-1 encoding: {ex}")
            return

    # Ensure the text column exists
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the input CSV file.")

    # Fill missing text with empty strings
    df[text_column] = df[text_column].fillna("").astype(str)

    # Detect language for each row
    df["Detected Language"] = df[text_column].apply(lambda x: detect_language(x, loaded_vectorizer, loaded_model))

    # DataFrames for Naive Bayes processing
    naive_bayes_tagalog = []
    naive_bayes_english = []
    naive_bayes_taglish = []

    # Function to decide if the text should go to Naive Bayes or BERT based on sentence count
    def process_sentiment(text, lang):
        sentences = split_into_sentences(text)
        if len(sentences) > 1:
            print(f"Text passed to BERT model (more than 1 sentence): {text}")
        else:
            print(f"Text passed to Naive Bayes model: {text}")
            if lang == "Tagalog":
                naive_bayes_tagalog.append(text)
            elif lang == "English":
                naive_bayes_english.append(text)
            elif lang == "Taglish":
                naive_bayes_taglish.append(text)

    # Apply the sentiment analysis function
    df.apply(lambda row: process_sentiment(row[text_column], row["Detected Language"]), axis=1)

    # Separate the rows based on detected language
    df_tagalog = df[df["Detected Language"] == "Tagalog"]
    df_english = df[df["Detected Language"] == "English"]
    df_taglish = df[df["Detected Language"] == "Taglish"]

    # Save the separated data to respective CSV files
    df_tagalog.to_csv(tagalog_csv, index=False, encoding="utf-8")
    df_english.to_csv(english_csv, index=False, encoding="utf-8")
    df_taglish.to_csv(taglish_csv, index=False, encoding="utf-8")

    # Save the Naive Bayes-specific CSV files
    pd.DataFrame({text_column: naive_bayes_tagalog}).to_csv(tagalognaivebayes_csv, index=False, encoding="utf-8")
    pd.DataFrame({text_column: naive_bayes_english}).to_csv(englishnaivebayes_csv, index=False, encoding="utf-8")
    pd.DataFrame({text_column: naive_bayes_taglish}).to_csv(taglishnaivebayes_csv, index=False, encoding="utf-8")

    print(f"Processed CSV files saved as:\n"
          f"- Tagalog (BERT): {tagalog_csv}\n- English (BERT): {english_csv}\n- Taglish (BERT): {taglish_csv}\n"
          f"- Tagalog (Naive Bayes): {tagalognaivebayes_csv}\n"
          f"- English (Naive Bayes): {englishnaivebayes_csv}\n"
          f"- Taglish (Naive Bayes): {taglishnaivebayes_csv}")

# Step 5: Main script
if __name__ == "__main__":
    # Specify input and output file paths
    input_csv = "base.csv"  # Path to the input CSV file
    tagalog_csv = "./BERT/BERT_sentiments/tagalog_sentences.csv"  # Path for Tagalog sentences
    english_csv = "./BERT/BERT_sentiments/english_sentences.csv"  # Path for English sentences
    taglish_csv = "./BERT/BERT_sentiments/taglish_sentences.csv"  # Path for Taglish sentences
    text_column = "text"  # Column name containing text to analyze

    # New directories for Naive Bayes
    tagalognaivebayes_csv = "./NaiveBayes/naive_bayes_sentiments/tagalog_sentences.csv"
    englishnaivebayes_csv = "./NaiveBayes/naive_bayes_sentiments/english_sentences.csv"
    taglishnaivebayes_csv = "./NaiveBayes/naive_bayes_sentiments/taglish_sentences.csv"

    # Ensure the Naive Bayes directories exist
    os.makedirs(os.path.dirname(tagalognaivebayes_csv), exist_ok=True)
    os.makedirs(os.path.dirname(englishnaivebayes_csv), exist_ok=True)
    os.makedirs(os.path.dirname(taglishnaivebayes_csv), exist_ok=True)

    try:
        process_csv(input_csv, tagalog_csv, english_csv, taglish_csv, text_column,
                    tagalognaivebayes_csv, englishnaivebayes_csv, taglishnaivebayes_csv)
    except Exception as e:
        print(f"Error: {e}")
