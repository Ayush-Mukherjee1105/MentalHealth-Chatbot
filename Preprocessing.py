import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv('Combined Data.csv')

# Rename columns properly
df = df.rename(columns={'statement': 'text', 'status': 'label'})

# Drop the unnamed column
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

print(df.head())
print(df['label'].value_counts())

# Plot distribution
plt.figure(figsize=(10,5))
sns.countplot(data=df, x='label', order=df['label'].value_counts().index)
plt.xticks(rotation=45)
plt.title('Distribution of Mental Health Labels')
plt.show()

# Preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if isinstance(text, float):
        text = ''
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

print("Sample Cleaned Text:")
print(df['clean_text'].head())

# Generate wordclouds for each label
labels = df['label'].unique()
for label in labels:
    subset = df[df['label'] == label]
    text_combined = " ".join(subset['clean_text'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_combined)
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud for {label}")
    plt.show()

# Train-Test Split
X = df['clean_text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save processed files
train_df = pd.DataFrame({"text": X_train, "label": y_train})
test_df = pd.DataFrame({"text": X_test, "label": y_test})

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

print("Preprocessing Complete! Training and Testing datasets are saved.")
