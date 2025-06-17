import os
import httpx
import asyncio
import warnings
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Suppress warnings
warnings.filterwarnings("ignore")

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment and model configuration
HF_API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

headers = {}

# Translator placeholder
translator = None

class Message(BaseModel):
    text: str

async def load_models():
    global translator
    try:
        from translatepy import Translator
        print("🔄 Initializing translator...")
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
        print(f"🔍 HF API status code: {response.status_code}")
        print(f"🔍 HF API response: {response.text[:200]}...")

        if response.status_code != 200:
            return False
        return True
    except Exception as e:
        print(f"❌ Exception during HF API test: {e}")
        traceback.print_exc()
        return False

@app.on_event("startup")
async def startup_event():
    global headers

    # Check if HF API token is set
    if not HF_API_TOKEN:
        print("❌ Environment variable HF_API_TOKEN not set. Aborting startup.")
        await asyncio.sleep(1)
        os._exit(1)

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }

    try:
        print("🚀 Starting application...")
        await load_models()
        hf_ready = await test_hf_api()

        if hf_ready:
            print("✅ Hugging Face API test passed.")
        else:
            print("⚠️ WARNING: Hugging Face API test failed. Check token, model name, or usage limits.")

        print("✅ Startup complete.")
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

        # Send input to Hugging Face inference API
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
        print(f"❌ Error during moderation: {e}")
        traceback.print_exc()
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
    
