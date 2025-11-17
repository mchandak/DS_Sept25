import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns
import re
from nltk.corpus import stopwords

# Download stopwords if not already present
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load the data
file_path = 'D:\\Manoj\\1ExcelR\\Data\\apple.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Text preprocessing
text = text.lower()
text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
tokens = [word for word in text.split() if word not in stop_words]

# Join tokens back to string
cleaned_text = ' '.join(tokens)
cleaned_text

# ---------------------------
# Word Cloud
# ---------------------------
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(cleaned_text)

# ---------------------------
# Bar Plot of Top 15 Words
# ---------------------------
word_counts = Counter(tokens)
top_words = word_counts.most_common(15)
top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])

# Display visualizations
fig, axs = plt.subplots(2, 1, figsize=(12, 18))

# Word Cloud
axs[0].imshow(wordcloud, interpolation='bilinear')
axs[0].axis('off')
axs[0].set_title('Word Cloud of Apple MacBook Reviews')

# Bar Plot - Top Words
sns.barplot(x='Frequency', y='Word', data=top_words_df, ax=axs[1])
axs[1].set_title('Top 15 Frequent Words')


