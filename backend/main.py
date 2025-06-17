import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from better_profanity import profanity
from translatepy import Translator
from transformers import pipeline

app = FastAPI()
translator = Translator()

# Load transformers hate speech model
hate_speech_detector = pipeline("text-classification", model="Hate-speech-CNERG/dehatebert-mono-english")

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    profanity.load_censor_words()

@app.post("/moderate")
async def moderate_message(message: Message):
    try:
        original_text = message.text.strip()

        # Detect & translate if needed
        detection = translator.detect(original_text)
        source_lang_code = detection.result.language_code.lower()
        translated_text = original_text

        if detection.result.language.lower() != "english":
            translated = translator.translate(original_text, "English")
            translated_text = translated.result

        # Profanity check
        is_profane = profanity.contains_profanity(translated_text)
        censored_text = profanity.censor(translated_text)

        # Hate speech detection
        hate_result = hate_speech_detector(translated_text)[0]
        hate_label = hate_result["label"].lower()
        hate_score = hate_result["score"]

        # Determine final status
        is_hateful = "hate" in hate_label or "offensive" in hate_label
        status = "inappropriate" if is_profane or is_hateful else "clean"

        return {
            "status": status,
            "original_text": original_text,
            "translated_text": translated_text,
            "censored_text": censored_text,
            "source_language": source_lang_code,
            "hate_label": hate_label,
            "hate_score": round(hate_score, 4),
            "profanity_detected": is_profane,
            "score": max(hate_score, 0.99 if is_profane else 0.95)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Moderation failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "ready",
        "version": "translate-profanity-hatespeech"
    }
    
