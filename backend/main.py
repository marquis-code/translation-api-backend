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

from dotenv import load_dotenv


# -------------------------------------------------------------------
# ENV / CONFIG
# -------------------------------------------------------------------
load_dotenv()  # loads backend/.env locally; Render uses environment variables

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
    allow_origins=ALLOW_ORIGINS,  # set via env: "https://domain.com,http://localhost:5173"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


# -------------------------------------------------------------------
# AUTH / PASSWORD HASHING
# -------------------------------------------------------------------
def hash_password(password: str) -> str:
    """
    Hash a password using PBKDF2.
    NOTE: For real production auth, use a proper user table + per-user salt.
    """
    salt = PASSWORD_SALT.encode("utf-8")
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000).hex()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return hmac.compare_digest(hash_password(plain_password), hashed_password)


def create_access_token(data: dict) -> str:
    """
    Create JWT access token.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Verify JWT from Authorization header.
    """
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
# Pre-hashed password for "password123" (DEMO ONLY)
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
    timestamp: float
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
    status: str  # "in_progress", "completed"
    created_at: datetime
    completed_at: Optional[datetime]


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
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password"
        )

    access_token = create_access_token(data={"sub": user_data.username})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"username": user["username"], "full_name": user["full_name"], "role": user["role"]},
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
    return {"consultation_id": consultation_id}


@app.get("/api/consultations/{consultation_id}")
async def get_consultation(consultation_id: str, username: str = Depends(verify_token)):
    consultation = consultations_db.get(consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")
    if consultation["doctor_id"] != username:
        raise HTTPException(status_code=403, detail="Not authorized")
    return consultation


@app.post("/api/consultations/{consultation_id}/summary")
async def generate_summary(consultation_id: str, username: str = Depends(verify_token)):
    consultation = consultations_db.get(consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    transcript_text = "\n".join([f"{seg['speaker']}: {seg['text']}" for seg in consultation["transcript"]])

    try:
        summary = generate_clinical_summary(transcript_text)
        consultation["summary"] = summary
        return {"summary": summary}
    except Exception:
        summary = {
            "identifiers": "Name: [To be filled]\nAge: [To be filled]\nSex: [To be filled]\nLocation: [To be filled]",
            "history": extract_section(transcript_text, "history"),
            "examination": extract_section(transcript_text, "examination"),
            "diagnoses": extract_section(transcript_text, "diagnoses"),
            "treatment": extract_section(transcript_text, "treatment"),
            "advice": extract_section(transcript_text, "advice"),
            "next_steps": "Follow-up appointment recommended in 2 weeks.",
        }
        consultation["summary"] = summary
        return {"summary": summary}


@app.put("/api/consultations/{consultation_id}/summary")
async def update_summary(
    consultation_id: str, summary: ClinicalSummary, username: str = Depends(verify_token)
):
    consultation = consultations_db.get(consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    consultation["summary"] = summary.dict()
    return {"message": "Summary updated successfully"}


@app.post("/api/consultations/{consultation_id}/complete")
async def complete_consultation(consultation_id: str, username: str = Depends(verify_token)):
    consultation = consultations_db.get(consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")

    consultation["status"] = "completed"
    consultation["completed_at"] = datetime.utcnow().isoformat()
    return {"message": "Consultation marked as completed"}


@app.get("/api/consultations")
async def list_consultations(username: str = Depends(verify_token)):
    return [c for c in consultations_db.values() if c["doctor_id"] == username]


# -------------------------------------------------------------------
# WEBSOCKET (MOCK STREAMING)
# -------------------------------------------------------------------
@app.websocket("/ws/transcribe/{consultation_id}")
async def websocket_transcribe(websocket: WebSocket, consultation_id: str):
    print(f"check 1")
    await websocket.accept()
    print(f"check b")
    consultation = consultations_db.get(consultation_id)
    if not consultation:
        print(f"no confam")
        await websocket.close(code=1008)
        return
    
    try:
        while True:
            # Receive audio bytes
            print(f"check 2")
            _data = await websocket.receive_bytes()
            print(_data)
            # TODO: Replace with real AssemblyAI streaming in streaming_service.py
            response = {
                "type": "transcript",
                "text": "Sample transcription text",
                "speaker": "Doctor",
                "timestamp": datetime.utcnow().isoformat(),
                "confidence": 0.95,
                "is_final": False,
            }
            print(f"check 3")
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print(f"WebSocket disconnected for consultation main {consultation_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1011)


# -------------------------------------------------------------------
# SUMMARY HELPERS
# -------------------------------------------------------------------
def generate_clinical_summary(transcript: str) -> dict:
    """
    Generate structured clinical summary from transcript.
    In production: use AssemblyAI LeMUR or an LLM provider.
    """
    return {
        "identifiers": "Name: Patient Name\nAge: [From transcript]\nSex: [From transcript]\nLocation: [From transcript]",
        "history": "Chief Complaint: Extracted from transcript\nHistory of Present Illness: Key details from patient's description",
        "examination": "Vital signs and physical examination findings mentioned in the consultation",
        "diagnoses": "Primary and secondary diagnoses based on the consultation",
        "treatment": "Prescribed medications and treatment plan",
        "advice": "Lifestyle modifications and patient counseling provided",
        "next_steps": "Follow-up appointments and investigations recommended",
    }


def extract_section(transcript: str, section: str) -> str:
    return f"Information extracted from transcript related to {section}. (Replace with actual NLP extraction)"


# -------------------------------------------------------------------
# LOCAL RUN (Render should use uvicorn start command)
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
