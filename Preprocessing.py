import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Create output folder if not exist
os.makedirs('outputs', exist_ok=True)

# 1. Load the Dataset
df = pd.read_csv('Combined Data.csv')

# 2. Check basic info
print("Dataset shape:", df.shape)
print(df.head())

# 3. Check for missing values
print(df.isnull().sum())

# 4. Drop missing data (if any)
df = df.dropna()

# 5. Visualize class distribution
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='label')
plt.title("Sentiment Class Distribution")
plt.xticks(rotation=45)
plt.savefig('outputs/class_distribution.png')
plt.show()

# 6. Encode Labels
le = LabelEncoder()
df['encoded_label'] = le.fit_transform(df['label'])

# Save Label Mapping
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Label Mapping:", label_mapping)

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], 
    df['encoded_label'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['encoded_label']
)

# Save processed files
train_data = pd.DataFrame({'text': X_train, 'label': y_train})
test_data = pd.DataFrame({'text': X_test, 'label': y_test})

train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)

print("Preprocessing complete. Files saved: train.csv, test.csv")
