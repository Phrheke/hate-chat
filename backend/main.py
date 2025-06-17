import os
import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from better_profanity import profanity
from translatepy import Translator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

# Global objects
translator = None
hate_model = None
vectorizer = None

@app.on_event("startup")
async def startup_event():
    global translator, hate_model, vectorizer
    try:
        profanity.load_censor_words()
        translator = Translator()

        # Load hate speech model
        model_path = "hate_model.pkl"
        vectorizer_path = "vectorizer.pkl"

        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model or vectorizer file not found")

        with open(model_path, "rb") as f:
            hate_model = pickle.load(f)

        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        print("Startup: models loaded successfully")

    except Exception as e:
        print(f"Startup error: {str(e)}")
        raise RuntimeError("Startup failed") from e

@app.post("/moderate")
async def moderate_message(message: Message):
    if not translator or not hate_model or not vectorizer:
        raise HTTPException(status_code=503, detail="Service unavailable - models not ready")

    try:
        original_text = message.text
        translation = translator.translate(original_text, "en").result
        lang = translator.language(original_text).result.alpha2

        # Profanity check
        is_profane = profanity.contains_profanity(translation)
        censored = profanity.censor(translation)

        # Hate speech detection
        X = vectorizer.transform([translation])
        prediction = hate_model.predict(X)[0]
        prob = hate_model.predict_proba(X)[0].max()

        status = "inappropriate" if is_profane or prediction == 1 else "clean"

        return {
            "status": status,
            "original_text": original_text,
            "translated_text": translation,
            "source_language": lang,
            "censored_text": censored,
            "hate_speech_detected": bool(prediction),
            "profanity_detected": is_profane,
            "confidence": round(prob, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Moderation failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "ready" if translator and hate_model and vectorizer else "loading",
        "version": "profanity+hate"
    }
    
    
