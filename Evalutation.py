# Evaluation.py

import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and tokenizer
model = TFBertForSequenceClassification.from_pretrained('saved_model')
tokenizer = BertTokenizer.from_pretrained('saved_model')

# Load test data
test_df = pd.read_csv('test.csv')

# Tokenize
def tokenize_data(texts, max_len=128):
    texts = list(texts)
    tokens = tokenizer(
        texts,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    return tokens

X_test_tokens = tokenize_data(test_df['text'])
y_test = test_df['label'].values

# Predict
logits = model.predict(X_test_tokens.data).logits
y_pred = np.argmax(logits, axis=1)

# Metrics
print("Classification Report:\n")
print(classification_report(y_test, y_pred, digits=4))

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('outputs/confusion_matrix.png')
plt.show()

print("\nEvaluation complete ✅")
