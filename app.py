# app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# Load model and tokenizer
model_path = 'saved_model'
model = TFBertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Label Mapping (Update if your labels are different)
label_mapping = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise"
}

# Function to predict
def predict_sentiment(text):
    inputs = tokenizer(
        text, 
        return_tensors="tf", 
        truncation=True, 
        padding=True, 
        max_length=128
    )
    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1)
    pred_class = np.argmax(probs, axis=1)[0]
    confidence = probs[0][pred_class].numpy()
    return label_mapping[pred_class], confidence

# Streamlit UI
st.set_page_config(page_title="AI Mental Health Chatbot", page_icon="💬", layout="centered")

st.title("💬 AI Mental Health Chatbot")
st.write("### Understand the emotional tone of your thoughts.")

# Input box
user_input = st.text_area("Enter your thoughts here 👇", height=150)

if st.button("Analyze Emotion"):
    if user_input.strip() != "":
        with st.spinner('Analyzing...'):
            label, confidence = predict_sentiment(user_input)
            st.success(f"**Detected Emotion:** {label.capitalize()} ({confidence*100:.2f}% confidence)")
            
            # Emojis for emotions
            emojis = {
                "anger": "😠",
                "fear": "😨",
                "joy": "😄",
                "love": "❤️",
                "sadness": "😢",
                "surprise": "😲"
            }
            st.markdown(f"# {emojis.get(label, '')}")
    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using BERT and Streamlit.")
