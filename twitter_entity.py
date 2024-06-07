import csv

def clean_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            try:
                writer.writerow(row)
            except Exception as e:
                print(f"Skipping line due to error: {e}")

clean_csv('/content/twitter_train.csv', '/content/cleaned_twitter_train.csv')
clean_csv('/content/twitter_validate.csv', '/content/cleaned_twitter_validate.csv')import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('stopwords')
# Load the cleaned datasets
train_df = pd.read_csv('/content/cleaned_twitter_train.csv')
test_df = pd.read_csv('/content/cleaned_twitter_validate.csv')
print("Train Dataset:")
print(train_df.head())
print("\nTest Dataset:")
print(test_df.head())
print("Train Dataset:")
print(train_df.tail())
print("\nTest Dataset:")
print(test_df.tail())
print(train_df.shape)
print(test_df.shape)
print("\nTrain Dataset Types:")
print(train_df.dtypes)
print("\nTest Dataset Types:")
print(test_df.dtypes)
def preprocess_text(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]
        tokens = [token for token in tokens if token not in stopwords.words('english')]
        tokens = [token for token in tokens if token.isalpha()]

        return ' '.join(tokens)
    else:
        return ""

train_df['cleaned_text'] = train_df['im getting on borderlands and i will murder you all ,'].apply(preprocess_text)
test_df['cleaned_text'] = test_df['I mentioned on Facebook that I was struggling for motivation to go for a run the other day, which has been translated by Tom’s great auntie as ‘Hayley can’t get out of bed’ and told to his grandma, who now thinks I’m a lazy, terrible person 🤣'].apply(preprocess_text)
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

train_df['sentiment'] = train_df['cleaned_text'].apply(get_sentiment)
test_df['sentiment'] = test_df['cleaned_text'].apply(get_sentiment)

def classify_sentiment(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

train_df['sentiment_label'] = train_df['sentiment'].apply(classify_sentiment)
test_df['sentiment_label'] = test_df['sentiment'].apply(classify_sentiment)
sns.set(style="darkgrid")

plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment_label', data=train_df, palette=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (Train Dataset)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(10, 6))
sns.countplot(x='sentiment_label', data=test_df, palette=['green', 'gray', 'red'])
plt.title('Sentiment Distribution (Test Dataset)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(12, 6))
sns.histplot(train_df['sentiment'], bins=30, kde=True, color='blue')
plt.title('Sentiment Scores Distribution (Train Dataset)')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(12, 6))
sns.histplot(test_df['sentiment'], bins=30, kde=True, color='blue')
plt.title('Sentiment Scores Distribution (Test Dataset)')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()
# Assuming 'Borderlands' and 'Facebook' columns represent topics or brands in both datasets
train_df['topic'] = train_df['Borderlands']
test_df['topic'] = test_df['Facebook']

plt.figure(figsize=(12, 6))
sns.countplot(x='sentiment_label', hue='topic', data=train_df, palette=['green', 'gray', 'red'])
plt.title('Sentiment Distribution by Topic (Train Dataset)')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.legend(title='Topic')
plt.show()

