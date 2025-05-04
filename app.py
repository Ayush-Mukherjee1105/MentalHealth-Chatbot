import os
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

# === Load API key and configure Gemini ===
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gen_model = genai.GenerativeModel("gemini-2.0-flash")

# === Setup logging ===
LOG_FILE = "conversation_log.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "user_input", "emotion", "chatbot_response"]).to_csv(LOG_FILE, index=False)

# === Streamlit UI ===
st.set_page_config(page_title="Mental Health Chatbot", layout="centered", initial_sidebar_state="collapsed")
st.title("🧠 Mental Health Chatbot")
st.markdown("Empathetic, context-aware conversations")

# Session state initialization
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.last_emotion = None
if "chat_model" not in st.session_state:
    st.session_state.chat_model = gen_model.start_chat(history=[])

# Chat input
user_input = st.chat_input("How are you feeling today?")
if user_input:
    try:
        # Prompt Gemini for emotion & response
        prompt = (
            f"The user said: '{user_input}'.\n"
            "First, determine the user's emotion using one word (e.g., sadness, joy, anger, fear, etc).\n"
            "Then, respond with empathy and compassion in a separate paragraph."
        )
        response = st.session_state.chat_model.send_message(prompt).text.strip()

        # Parse response: Assume format "[emotion]\n\n[response]"
        parts = response.split("\n\n", 1)
        detected_emotion = parts[0].strip().capitalize()
        gemini_response = parts[1].strip() if len(parts) > 1 else "[Could not parse response]"

        # Log to file
        timestamp = datetime.now().isoformat()
        log_df = pd.DataFrame([[timestamp, user_input, detected_emotion, gemini_response]],
                              columns=["timestamp", "user_input", "emotion", "chatbot_response"])
        log_df.to_csv(LOG_FILE, mode="a", header=False, index=False)

        # Save to history
        st.session_state.history.append((user_input, gemini_response, detected_emotion))

    except Exception as e:
        st.session_state.history.append((user_input, f"⚠️ Error: {e}", "N/A"))

# Display chat
for user_msg, bot_msg, emotion in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
        # Only show emotion if it changed from the last message
        if emotion != st.session_state.last_emotion:
            st.caption(f"🧠 Detected Emotion: {emotion}")
            st.session_state.last_emotion = emotion
