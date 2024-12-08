import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the CSV file
csv_path = './final_output.csv'
df = pd.read_csv(csv_path)

# Move the contents of 'Predicted Sentiment' column to 'sentiment', if present, and delete it
if 'Predicted Sentiment' in df.columns:
    df['sentiment'] = df['Predicted Sentiment']
    df = df.drop(columns=['Predicted Sentiment'])

# Convert the 'sentiment' column to lowercase after reading the CSV
df['sentiment'] = df['sentiment'].str.lower()

# Drop rows where either 'date' or 'sentiment' is NaN
df = df.dropna(subset=['date', 'sentiment'])

# Convert the 'date' column from 'dd/mm/yy' to datetime format
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%y', errors='coerce')

# Map sentiment to numerical values (positive=7, neutral=5, negative=3)
sentiment_map = {'positive': 7, 'neutral': 5, 'negative': 3}
df['sentiment'] = df['sentiment'].map(sentiment_map)

# Drop rows where 'sentiment' couldn't be mapped (NaN values after mapping)
df = df.dropna(subset=['sentiment'])

# Create a new DataFrame with the correct column names for Prophet
df = df.rename(columns={'date': 'ds', 'sentiment': 'y'})  # Rename columns for Prophet

# Ensure 'ds' is in datetime format
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')

# Drop rows where 'ds' conversion failed (NaT values)
df = df.dropna(subset=['ds'])

# Sort the DataFrame by 'ds' to ensure chronological order
df = df.sort_values('ds')

# Initialize the Prophet model
model = Prophet()

# Fit the model to the data
model.fit(df)

# Make future predictions (predict the next 5 months)
future = model.make_future_dataframe(periods=5, freq='M')

# Predict the future trends
forecast = model.predict(future)

# Snap the forecasted values ('yhat') to the nearest valid sentiment score
valid_sentiments = [3, 5, 7]
forecast['yhat_snapped'] = forecast['yhat'].apply(lambda x: min(valid_sentiments, key=lambda val: abs(x - val)))

# Combine historical and forecast data for consistent x-axis labels
combined_df = pd.concat([df[['ds', 'y']], forecast[['ds', 'yhat_snapped']].rename(columns={'yhat_snapped': 'y'})], ignore_index=True)

# Extract unique months from combined DataFrame for x-axis labels
combined_df['month'] = combined_df['ds'].dt.to_period('M')  # Convert to monthly period
unique_months = combined_df.drop_duplicates('month')  # Keep only one row per month

# Plotting
fig, ax = plt.subplots(figsize=(12, 7))

# Plot historical data in blue
ax.plot(df['ds'], df['y'], marker='o', color='blue', label='Historical Data', linewidth=2)

# Plot forecast data in red with enhanced styling
ax.plot(
    forecast[forecast['ds'] > df['ds'].max()]['ds'], 
    forecast[forecast['ds'] > df['ds'].max()]['yhat_snapped'], 
    marker='o', color='red', linestyle='--', linewidth=2, label='Forecast (Snapped)', alpha=0.8
)

# Add a vertical line to separate historical and forecast data
ax.axvline(x=df['ds'].max(), color='gray', linestyle='--', linewidth=1.5)

# Add background text for the forecast section
x_max = forecast['ds'].max()
x_min = forecast[forecast['ds'] > df['ds'].max()]['ds'].min()
y_min, y_max = ax.get_ylim()
ax.text(
    x=(x_min + (x_max - x_min) / 2), 
    y=y_min + (y_max - y_min) / 2, 
    s="Predicted Sentiments in the Future", 
    fontsize=20, color='lightgray', alpha=0.5, ha='center', va='center', rotation=45
)

# Customize the x-axis ticks and labels to show only one label per month
ax.set_xticks(unique_months['ds'])
ax.set_xticklabels(unique_months['ds'].dt.strftime('%Y-%m'), rotation=45, fontsize=10)

# Customize the y-axis to display sentiment levels with labels
ax.set_yticks([3, 5, 7])  # Set y-axis ticks
ax.set_yticklabels(['Negative', 'Neutral', 'Positive'], fontsize=10)  # Replace ticks with labels

# Add title, labels, legend, and show plot
plt.title('Sentiment Trend Prediction (Snapped)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sentiment Category', fontsize=12)
plt.legend(fontsize=10)
plt.tight_layout()  # Adjust layout to prevent clipping
plt.grid(alpha=0.3)  # Add light gridlines for better visual appeal
plot_path = './static/sentiment_trend_forecast.png'
plt.savefig(plot_path, dpi=300)


# Display the forecasted results with labels
forecast['predicted_sentiment'] = forecast['yhat_snapped'].apply(lambda x: 'Positive' if x == 7 else 'Neutral' if x == 5 else 'Negative')

# Print the forecasted results with sentiment categories
print(forecast[['ds', 'yhat', 'yhat_snapped', 'yhat_lower', 'yhat_upper', 'predicted_sentiment']])
