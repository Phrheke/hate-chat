import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from better_profanity import profanity
from translatepy import Translator

app = FastAPI()
translator = Translator()
hate_model = None
vectorizer = None

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
    global hate_model, vectorizer
    profanity.load_censor_words()
    try:
        hate_model = joblib.load("hate_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
    except Exception as e:
        print("Failed to load hate speech model:", e)
        os._exit(1)

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

        # Hate speech detection (scikit-learn)
        X = vectorizer.transform([translated_text])
        hate_label = hate_model.predict(X)[0]
        hate_prob = max(hate_model.predict_proba(X)[0])

        # Final moderation status
        is_hateful = hate_label.lower() in ["hate", "offensive"]
        status = "inappropriate" if is_profane or is_hateful else "clean"

        return {
            "status": status,
            "original_text": original_text,
            "translated_text": translated_text,
            "censored_text": censored_text,
            "source_language": source_lang_code,
            "hate_label": hate_label,
            "hate_score": round(hate_prob, 4),
            "profanity_detected": is_profane
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Moderation failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "ready" if hate_model and vectorizer else "loading",
        "version": "lite-ml"
    }
    
