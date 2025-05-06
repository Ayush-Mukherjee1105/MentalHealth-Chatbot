import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import re
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load your dataset
file_path = 'ChatbotWebsite/static/data/empatheticdialogues.csv'  
df = pd.read_csv(file_path)

# Basic checks
df['Situation'] = df['Situation'].fillna('')
df['emotion'] = df['emotion'].fillna('unknown')

# NLP setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# Clean text
df['clean_text'] = df['Situation'].apply(preprocess)

# Filter to top 32 emotions
emotion_counts = df['emotion'].value_counts()
top_32_emotions = emotion_counts.head(32).index.tolist()
df = df[df['emotion'].isin(top_32_emotions)]

# WORDCLOUDS
for emotion in top_32_emotions:
    text = " ".join(df[df['emotion'] == emotion]['clean_text'])
    if text.strip():  # Skip empty
        wc = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"WordCloud for: {emotion}")
        plt.tight_layout()
        plt.show()

# EMOTION DISTRIBUTION
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='emotion', order=top_32_emotions, palette='viridis')
plt.title("Distribution of Top 32 Emotions")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# CONFUSION MATRIX via simple classifier
X = df['clean_text']
y = df['emotion']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

# Simple classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
fig, ax = plt.subplots(figsize=(14, 14))
disp.plot(xticks_rotation=90, ax=ax, cmap='Blues')
plt.title("Confusion Matrix of Emotions (Logistic Regression)")
plt.tight_layout()
plt.show()
