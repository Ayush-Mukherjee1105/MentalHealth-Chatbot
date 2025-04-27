# ModelTraining.py

import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Function to force GPU usage
def force_gpu_usage():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to avoid allocation issues
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Running on GPU: {gpus}")
        except RuntimeError as e:
            print(f"❌ GPU setup failed: {e}")
    else:
        print("❌ No GPU found. Running on CPU.")

# Call the function to force GPU usage
force_gpu_usage()
# Load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Correct tokenize function
def tokenize_data(texts, labels, max_len=128):
    texts = texts.astype(str).tolist()  # Ensure list of strings
    labels = labels.tolist()

    encodings = tokenizer.batch_encode_plus(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='tf'
    )
    return encodings, tf.convert_to_tensor(labels)

# Tokenize
X_train_tokens, y_train = tokenize_data(train_df['text'], train_df['label'])
X_test_tokens, y_test = tokenize_data(test_df['text'], test_df['label'])

# Load model
num_classes = len(train_df['label'].unique())
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# Compile
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = ['accuracy']

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train
history = model.fit(
    X_train_tokens,
    y_train,
    validation_data=(X_test_tokens, y_test),
    epochs=4,
    batch_size=16
)

# Save model
os.makedirs('saved_model', exist_ok=True)
model.save_pretrained('saved_model')
tokenizer.save_pretrained('saved_model')

print("✅ Model saved!")

# Save graphs
os.makedirs('outputs', exist_ok=True)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.savefig('outputs/metrics.png')
plt.show()
