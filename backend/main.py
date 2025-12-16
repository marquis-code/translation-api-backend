from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict
import jwt
import assemblyai as aai
from datetime import datetime, timedelta
import os
import hashlib
import hmac
import uuid
import asyncio
import logging

from dotenv import load_dotenv
from streaming_service import TranscriptionService, SummaryService
from websocket_manager import manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# ENV / CONFIG
# -------------------------------------------------------------------
load_dotenv()

ENV = os.getenv("ENV", "development")
SECRET_KEY = os.getenv("SECRET_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "")
ALLOW_ORIGINS = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
PASSWORD_SALT = os.getenv("PASSWORD_SALT", "")

# Fail fast if critical secrets are missing
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY is missing. Set it in environment variables.")
if not ASSEMBLYAI_API_KEY:
    raise RuntimeError("ASSEMBLYAI_API_KEY is missing. Set it in environment variables.")
if not PASSWORD_SALT:
    raise RuntimeError("PASSWORD_SALT is missing. Set it in environment variables.")

# AssemblyAI config
aai.settings.api_key = ASSEMBLYAI_API_KEY

# -------------------------------------------------------------------
# FASTAPI APP
# -------------------------------------------------------------------
app = FastAPI(
    title="Medical Transcription API",
    docs_url=None if ENV == "production" else "/docs",
    redoc_url=None if ENV == "production" else "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS if ALLOW_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# -------------------------------------------------------------------
# AUTH / PASSWORD HASHING
# -------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Hash a password using PBKDF2."""
    salt = PASSWORD_SALT.encode("utf-8")
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000).hex()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return hmac.compare_digest(hash_password(plain_password), hashed_password)


def create_access_token(data: dict) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT from Authorization header."""
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# -------------------------------------------------------------------
# DEMO DATA (replace with DB in production)
# -------------------------------------------------------------------
DEMO_PASSWORD_HASH = hash_password("password123")

users_db = {
    "doctor@clinic.com": {
        "username": "doctor@clinic.com",
        "full_name": "Dr. Sarah Johnson",
        "hashed_password": DEMO_PASSWORD_HASH,
        "role": "doctor",
    }
}

consultations_db: Dict[str, dict] = {}
transcription_services: Dict[str, TranscriptionService] = {}

# -------------------------------------------------------------------
# MODELS
# -------------------------------------------------------------------
class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict


class TranscriptionSegment(BaseModel):
    text: str
    speaker: str
    timestamp: str
    confidence: float


class ClinicalSummary(BaseModel):
    identifiers: str
    history: str
    examination: str
    diagnoses: str
    treatment: str
    advice: str
    next_steps: str


class Consultation(BaseModel):
    id: str
    doctor_id: str
    transcript: List[TranscriptionSegment]
    summary: Optional[ClinicalSummary]
    status: str
    created_at: datetime
    completed_at: Optional[datetime]


class TranslationRequest(BaseModel):
    text: str
    target_language: str  # 'hi', 'en', 'ta', 'id'


# -------------------------------------------------------------------
# ROUTES
# -------------------------------------------------------------------
@app.get("/")
async def root():
    return {"message": "Medical Transcription API", "version": "1.0.0", "docs": "/docs"}


@app.post("/api/auth/login", response_model=Token)
async def login(user_data: UserLogin):
    user = users_db.get(user_data.username)
    if not user or not verify_password(user_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Incorrect username or password"
        )

    access_token = create_access_token(data={"sub": user_data.username})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "username": user["username"], 
            "full_name": user["full_name"], 
            "role": user["role"]
        },
    }


@app.post("/api/consultations/start")
async def start_consultation(username: str = Depends(verify_token)):
    consultation_id = str(uuid.uuid4())
    consultation = {
        "id": consultation_id,
        "doctor_id": username,
        "transcript": [],
        "summary": None,
        "status": "in_progress",
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
    }
    consultations_db[consultation_id] = consultation
    logger.info(f"Started consultation {consultation_id} for {username}")
    return {"consultation_id": consultation_id}


@app.get("/api/consultations/{consultation_id}")
async def get_consultation(consultation_id: str, username: str = Depends(verify_token)):
    consultation = consultations_db.get(consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")
    if consultation["doctor_id"] != username:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    # Get latest transcript from manager
    transcript = manager.get_transcript(consultation_id)
    if transcript:
        consultation["transcript"] = transcript
    
    return consultation


@app.post("/api/consultations/{consultation_id}/summary")
async def generate_summary(consultation_id: str, username: str = Depends(verify_token)):
    consultation = consultations_db.get(consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    # Get transcript from manager
    transcript_segments = manager.get_transcript(consultation_id)
    
    if not transcript_segments:
        raise HTTPException(status_code=400, detail="No transcript available")

    try:
        summary_service = SummaryService(ASSEMBLYAI_API_KEY)
        summary = await summary_service.generate_summary("", transcript_segments)
        consultation["summary"] = summary
        logger.info(f"Generated summary for consultation {consultation_id}")
        return {"summary": summary}
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@app.put("/api/consultations/{consultation_id}/summary")
async def update_summary(
    consultation_id: str, 
    summary: ClinicalSummary, 
    username: str = Depends(verify_token)
):
    consultation = consultations_db.get(consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    consultation["summary"] = summary.dict()
    logger.info(f"Updated summary for consultation {consultation_id}")
    return {"message": "Summary updated successfully"}


@app.post("/api/consultations/{consultation_id}/complete")
async def complete_consultation(consultation_id: str, username: str = Depends(verify_token)):
    consultation = consultations_db.get(consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    consultation["status"] = "completed"
    consultation["completed_at"] = datetime.utcnow().isoformat()
    
    # Cleanup
    if consultation_id in transcription_services:
        transcription_services[consultation_id].stop()
        del transcription_services[consultation_id]
    
    manager.clear_transcript(consultation_id)
    
    logger.info(f"Completed consultation {consultation_id}")
    return {"message": "Consultation marked as completed"}


@app.get("/api/consultations")
async def list_consultations(username: str = Depends(verify_token)):
    return [c for c in consultations_db.values() if c["doctor_id"] == username]


@app.post("/api/translate")
async def translate_text(request: TranslationRequest, username: str = Depends(verify_token)):
    """Translate text to target language using deep-translator"""
    try:
        from deep_translator import GoogleTranslator
        
        # Map language codes
        lang_map = {
            'hi': 'hi',  # Hindi
            'en': 'en',  # English
            'ta': 'ta',  # Tamil
            'id': 'id',  # Indonesian (Bahasa)
        }
        
        target = lang_map.get(request.target_language, 'en')
        
        translator = GoogleTranslator(source='auto', target=target)
        translated = translator.translate(request.text)
        
        return {
            "original": request.text,
            "translated": translated,
            "target_language": request.target_language
        }
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")


# -------------------------------------------------------------------
# WEBSOCKET FOR REAL-TIME TRANSCRIPTION - FIXED VERSION
# -------------------------------------------------------------------
@app.websocket("/ws/transcribe/{consultation_id}")
async def websocket_transcribe(websocket: WebSocket, consultation_id: str):
    """Handle WebSocket connection for real-time audio streaming and transcription"""
    
    # Connect WebSocket
    await manager.connect(consultation_id, websocket)
    
    consultation = consultations_db.get(consultation_id)
    if not consultation:
        await manager.broadcast_status(
            consultation_id, 
            "error", 
            "Consultation not found"
        )
        await websocket.close(code=1008)
        return
    
    # Initialize transcription service
    transcription_service = TranscriptionService(ASSEMBLYAI_API_KEY, consultation_id)
    transcription_services[consultation_id] = transcription_service
    
    # CRITICAL FIX: Get the current event loop BEFORE defining the callback
    loop = asyncio.get_event_loop()
    
    # Define callback for transcription updates
    def on_transcript(transcript_data):
        """
        Callback to handle transcription updates.
        This runs in a different thread, so we use run_coroutine_threadsafe
        to safely schedule the async operation in the main event loop.
        """
        try:
            # Schedule the coroutine in the main event loop
            future = asyncio.run_coroutine_threadsafe(
                manager.send_transcript(consultation_id, transcript_data),
                loop
            )
            # Optional: wait for completion with timeout
            future.result(timeout=5.0)
        except Exception as e:
            logger.error(f"❌ Error in transcript callback: {e}")
    
    # Start transcription service
    try:
        transcription_service.start(on_transcript)
        await manager.broadcast_status(consultation_id, "connected", "Transcription started")
        logger.info(f"✅ Transcription service started for {consultation_id}")
        
        # Receive and process audio data
        while True:
            try:
                # Receive audio bytes from client
                audio_data = await websocket.receive_bytes()
                
                # Send audio to transcription service
                transcription_service.send_audio(audio_data)
                
            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket disconnected for consultation {consultation_id}")
                break
            except Exception as e:
                logger.error(f"❌ Error processing audio: {e}")
                await manager.broadcast_status(
                    consultation_id, 
                    "error", 
                    f"Error processing audio: {str(e)}"
                )
                break
    
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        await manager.broadcast_status(
            consultation_id, 
            "error", 
            f"Transcription error: {str(e)}"
        )
    
    finally:
        # Cleanup
        logger.info(f"🧹 Cleaning up transcription service for {consultation_id}")
        transcription_service.stop()
        if consultation_id in transcription_services:
            del transcription_services[consultation_id]
        manager.disconnect(consultation_id)
        logger.info(f"✅ Cleanup complete for {consultation_id}")


# -------------------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_consultations": len(consultations_db),
        "active_transcriptions": len(transcription_services)
    }


# -------------------------------------------------------------------
# LOCAL RUN
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)