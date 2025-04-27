import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder if not exist
os.makedirs('outputs', exist_ok=True)

# 1. Load preprocessed data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 3. Tokenize datasets
def tokenize_data(texts, labels, max_len=128):
    tokens = tokenizer(
        list(texts),
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    return tokens, tf.convert_to_tensor(labels)

X_train_tokens, y_train = tokenize_data(train_df['text'], train_df['label'])
X_test_tokens, y_test = tokenize_data(test_df['text'], test_df['label'])

# 4. Load Pretrained BERT Model
num_classes = len(train_df['label'].unique())
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# 5. Compile Model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# 6. Train Model
history = model.fit(
    X_train_tokens.data,
    y_train,
    validation_data=(X_test_tokens.data, y_test),
    epochs=4,
    batch_size=16
)

# 7. Save Model
model.save_pretrained('saved_model')
tokenizer.save_pretrained('saved_model')
print("Model saved inside /saved_model folder.")

# 8. Plot Training History
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.savefig('outputs/metrics.png')
plt.show()

# 9. Evaluation
y_pred_logits = model.predict(X_test_tokens.data).logits
y_pred = np.argmax(y_pred_logits, axis=1)

print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('outputs/confusion_matrix.png')
plt.show()
