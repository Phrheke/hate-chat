import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import warnings
import asyncio

warnings.filterwarnings("ignore")

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str

translator = None
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}"
}

async def load_models():
    global translator
    try:
        from translatepy import Translator
        print("Initializing translator...")
        translator = Translator()
        print("Translator initialized.")
    except Exception as e:
        print(f"Error loading translator: {str(e)}")
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
    if not translator:
        raise HTTPException(status_code=503, detail="Service unavailable - translator not ready")
    
    try:
        # Translate to English
        translation = translator.translate(message.text, 'en').result
        lang = translator.language(message.text).result.alpha2

        # Send to Hugging Face Inference API
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": translation},
            timeout=10
        )
        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"HF API error: {response.text}")
        
        result = response.json()[0]  # First prediction

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
        "status": "ready" if translator else "loading",
        "version": "1.0.0"
    }
    
