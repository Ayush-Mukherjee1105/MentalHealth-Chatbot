# api/index.py
import os
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Load .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gen_model = genai.GenerativeModel("gemini-pro")

# Load emotion classifier
classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=False)

# FastAPI app
app = FastAPI()

# Allow frontend calls (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Log file setup
LOG_FILE = "conversation_log.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["timestamp", "user_input", "emotion", "chatbot_response"]).to_csv(LOG_FILE, index=False)

# Schema
class ChatRequest(BaseModel):
    message: str

# Chat Endpoint
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        message = request.message

        # Detect emotion
        emotion_result = classifier(message)
        detected_emotion = emotion_result[0]['label']

        # Gemini response
        prompt = (
            "You are a compassionate mental health support chatbot.\n"
            f"The user is feeling '{detected_emotion}' and said: '{message}'\n"
            "Respond with understanding and empathy."
        )
        gemini_response = gen_model.generate_content(prompt).text.strip()

        # Log
        timestamp = datetime.now().isoformat()
        log_entry = pd.DataFrame([[timestamp, message, detected_emotion, gemini_response]],
                                 columns=["timestamp", "user_input", "emotion", "chatbot_response"])
        log_entry.to_csv(LOG_FILE, mode="a", header=False, index=False)

        return {
            "response": gemini_response,
            "detected_emotion": detected_emotion
        }

    except Exception as e:
        return {"error": str(e)}

# Mount static frontend
app.mount("/static", StaticFiles(directory="../static"), name="static")

# Serve index.html at "/"
@app.get("/")
async def serve_frontend():
    return FileResponse("../static/index.html")
