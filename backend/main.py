import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from better_profanity import profanity

app = FastAPI()

# CORS configuration
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
        text = message.text
        is_profane = profanity.contains_profanity(text)
        censored = profanity.censor(text)

        return {
            "status": "inappropriate" if is_profane else "clean",
            "censored_text": censored,
            "original_text": text,
            "score": 1.0 if is_profane else 0.99,  # Dummy confidence score
            "source_language": "en",               # Assume English for now
            "translated_text": text                # No translation applied
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Moderation failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {
        "status": "ready",
        "version": "lite-profanity"
    }
    
