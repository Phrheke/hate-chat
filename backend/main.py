import os
import httpx
import traceback
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
headers = {}
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

async def load_models():
    global translator
    try:
        from translatepy import Translator
        print("🌐 Initializing translator...")
        translator = Translator()
        print("✅ Translator initialized.")
    except Exception as e:
        print(f"❌ Error loading translator: {str(e)}")
        raise

async def test_hf_api():
    print("🔌 Testing Hugging Face API connectivity...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": "Test message for sentiment analysis"},
                timeout=15
            )
        if response.status_code == 200:
            print("✅ Hugging Face API test successful:", response.json())
        else:
            print(f"⚠️ HF API returned status {response.status_code}: {response.text}")
            raise RuntimeError("Hugging Face API test failed")
    except Exception as e:
        print(f"❌ Error testing Hugging Face API: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    global headers

    hf_token = os.getenv("HF_API_TOKEN")
    if not hf_token:
        print("❌ Environment variable HF_API_TOKEN not set. Aborting startup.")
        await asyncio.sleep(1)
        os._exit(1)

    headers = {
        "Authorization": f"Bearer {hf_token}"
    }

    try:
        print("🚀 Starting service...")
        await load_models()
        await test_hf_api()
        print("✅ Startup complete. Service is ready.")
    except Exception:
        print("🔥 Fatal error during startup:")
        traceback.print_exc()
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

        # Call Hugging Face API
        async with httpx.AsyncClient() as client:
            response = await client.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": translation},
                timeout=15
            )

        if response.status_code != 200:
            raise HTTPException(status_code=502, detail=f"HF API error: {response.text}")

        result_data = response.json()
        if not result_data or not isinstance(result_data, list):
            raise HTTPException(status_code=500, detail="Invalid response format from HF API")

        result = result_data[0]

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
    
