import os
# Load data
os.chdir("D:\\Manoj\\1ExcelR\\Data")

# Open and read the contents of the file
with open('apple.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

# Display or process the text data as needed
print(text_data)

import nltk
#nltk.download("all")
import pandas as pd

# Split the text data into lines
lines = text_data.split('\n')

# Create a DataFrame with one column called "text"
df = pd.DataFrame({"text": lines})
df.info()

odd_index_data = df.iloc[1::2]
print(odd_index_data)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

# Instantiate the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Create a new column for sentiment
df['Sentiment_0'] = df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
df.head()

# Classify each statement as positive or negative based on the sentiment score
df['Sentiment'] = df['Sentiment_0'].apply(lambda x: 'positive' if x > 0 else 'negative')
df["Sentiment"].value_counts()

