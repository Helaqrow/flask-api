import pandas as pd
import os

# Function to clear or delete the base.csv file if it exists
def clear_base_csv():
    if os.path.exists('base.csv'):
        os.remove('base.csv')  # Delete the file if it exists
        print("Existing 'base.csv' has been deleted.")
    else:
        print("No existing 'base.csv' file found.")

# Function to search and collect matching rows from the CSV file
def search_and_collect(input_text):
    # Load the CSV file with a specified encoding
    csv_file = 'post_data.csv'
    try:
        df = pd.read_csv(csv_file, encoding='ISO-8859-1')  # Try using ISO-8859-1 encoding
    except UnicodeDecodeError as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Ensure the 'text' and 'date' columns exist in the CSV
    if 'text' not in df.columns or 'date' not in df.columns:
        print("Error: 'text' or 'date' column not found in the CSV file.")
        return

    # Filter rows where the 'text' column matches the user input (case-insensitive)
    matching_rows = df[df['text'].str.contains(input_text, case=False, na=False)]

    # Check if any matching rows were found
    if matching_rows.empty:
        print(f"No matches found for the input: {input_text}")
    else:
        # Select the relevant columns: 'text' and 'date'
        selected_data = matching_rows[['text', 'date']]

        # Add a new 'sentiment' column with empty values (as placeholder)
        selected_data['sentiment'] = ''

        # Save the collected data to a new CSV file
        selected_data.to_csv('base.csv', index=False)
        print(f"Collected data has been saved to 'base.csv'.")

# Main script
if __name__ == '__main__':
    # Clear or delete base.csv if it already exists
    clear_base_csv()

    # Prompt user for input text
    user_input = input("Enter the text to search for: ")
    search_and_collect(user_input)
