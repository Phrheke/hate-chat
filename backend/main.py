import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import warnings
import asyncio

# Suppress warnings
warnings.filterwarnings("ignore")

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production change to your frontend URL
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

# Global model variables
translator = None
classifier = None

async def load_models():
    global translator, classifier
    try:
        from translatepy import Translator
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

        print("Loading translation model...")
        translator = Translator()

        print("Loading moderation model...")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

        classifier = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1  # Ensures CPU use
        )

        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    try:
        await load_models()
    except Exception as e:
        print(f"Fatal error during startup: {str(e)}")
        await asyncio.sleep(1)
        os._exit(1)

@app.post("/moderate")
async def moderate_message(message: Message):
    if not translator or not classifier:
        raise HTTPException(status_code=503, detail="Service unavailable - models not loaded")
    
    try:
        # Translate to English
        translation = translator.translate(message.text, 'en').result
        lang = translator.language(message.text).result.alpha2

        # Moderate content
        result = classifier(translation, truncation=True, max_length=512)[0]

        return {
            "status": "inappropriate" if result['label'] == "NEGATIVE" else "clean",
            "score": result['score'],
            "label": result['label'],
            "translated_text": translation,
            "source_language": lang,
            "original_text": message.text
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Moderation failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "ready" if translator and classifier else "loading",
        "version": "1.0.0"
    }
    
