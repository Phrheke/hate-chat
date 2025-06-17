import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from better_profanity import profanity
from translatepy import Translator

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

translator = Translator()

class Message(BaseModel):
    text: str

# Load models on startup
@app.on_event("startup")
async def startup_event():
    global hate_model
    profanity.load_censor_words()
    hate_model = joblib.load("hate_speech_model_balanced.joblib")

@app.post("/moderate")
async def moderate_message(message: Message):
    try:
        original_text = message.text

        # Detect language using translator (correct usage)
        detection_result = translator.detect(original_text)
        source_lang = detection_result.language.lower()

        if source_lang != "english":
            translated = translator.translate(original_text, "en")
            text_for_analysis = translated.result
        else:
            text_for_analysis = original_text
            translated = None

        # Predict using your hate speech model
        prediction = hate_model.predict([text_for_analysis])[0]
        score = max(hate_model.predict_proba([text_for_analysis])[0])

        result = {
            "status": "inappropriate" if prediction == 1 else "clean",
            "score": round(score, 4),
            "source_language": source_lang,
            "translated_text": translated.result if translated else ""
        }
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Moderation failed: {str(e)}")
        
@app.get("/health")
async def health_check():
    return {"status": "ready", "version": "hate-speech+profanity"}
    
