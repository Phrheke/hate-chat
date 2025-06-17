import os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from translatepy import Translator

app = FastAPI()
translator = Translator()

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

class Message(BaseModel):
    text: str

@app.post("/moderate")
async def moderate_message(message: Message):
    try:
        translated = translator.translate(message.text, 'en').result
        lang = translator.language(message.text).result.alpha2

        async with httpx.AsyncClient() as client:
            response = await client.post(
                HF_API_URL,
                headers=HF_HEADERS,
                json={"inputs": translated}
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()[0]  # Top prediction

        return {
            "status": "inappropriate" if result['label'] == "NEGATIVE" else "clean",
            "score": result['score'],
            "label": result['label'],
            "translated_text": translated,
            "source_language": lang,
            "original_text": message.text
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ready", "version": "1.0.0"}
    
