from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict
import asyncio
import json
import jwt
import assemblyai as aai
from datetime import datetime, timedelta
import os
import hashlib
import hmac
import uuid

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours
ASSEMBLYAI_API_KEY = "7f9a4eb77c1d4fc1bfb13561a7ef3f14"

aai.settings.api_key = ASSEMBLYAI_API_KEY

# Initialize FastAPI
app = FastAPI(title="Medical Transcription API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# Simple password hashing using PBKDF2
def hash_password(password: str) -> str:
    """Hash a password using PBKDF2"""
    salt = b'medical_transcription_salt'  # In production, use unique salt per user
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000).hex()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return hmac.compare_digest(hash_password(plain_password), hashed_password)

# Pre-hashed password for "password123"
DEMO_PASSWORD_HASH = hash_password("password123")

# In-memory storage (replace with database in production)
users_db = {
    "doctor@clinic.com": {
        "username": "doctor@clinic.com",
        "full_name": "Dr. Sarah Johnson",
        "hashed_password": DEMO_PASSWORD_HASH,
        "role": "doctor"
    }
}

consultations_db = {}

# Models
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

# Helper Functions
def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Medical Transcription API",
        "version": "1.0.0",
        "docs": "/docs"
    }

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
        }
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
        "completed_at": None
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
async def generate_summary(
    consultation_id: str,
    username: str = Depends(verify_token)
):
    consultation = consultations_db.get(consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")
    
    # Generate summary using GPT-like prompt structure
    transcript_text = "\n".join([
        f"{seg['speaker']}: {seg['text']}" 
        for seg in consultation["transcript"]
    ])
    
    # Use AssemblyAI's LeMUR for summarization
    try:
        # For demo purposes, create a structured summary
        summary = generate_clinical_summary(transcript_text)
        consultation["summary"] = summary
        return {"summary": summary}
    except Exception as e:
        # Fallback to basic extraction
        summary = {
            "identifiers": "Name: [To be filled]\nAge: [To be filled]\nSex: [To be filled]\nLocation: [To be filled]",
            "history": extract_section(transcript_text, "history"),
            "examination": extract_section(transcript_text, "examination"),
            "diagnoses": extract_section(transcript_text, "diagnoses"),
            "treatment": extract_section(transcript_text, "treatment"),
            "advice": extract_section(transcript_text, "advice"),
            "next_steps": "Follow-up appointment recommended in 2 weeks."
        }
        consultation["summary"] = summary
        return {"summary": summary}

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
    return {"message": "Summary updated successfully"}

@app.post("/api/consultations/{consultation_id}/complete")
async def complete_consultation(
    consultation_id: str,
    username: str = Depends(verify_token)
):
    consultation = consultations_db.get(consultation_id)
    if not consultation:
        raise HTTPException(status_code=404, detail="Consultation not found")
    
    consultation["status"] = "completed"
    consultation["completed_at"] = datetime.utcnow().isoformat()
    return {"message": "Consultation marked as completed"}

@app.get("/api/consultations")
async def list_consultations(username: str = Depends(verify_token)):
    user_consultations = [
        c for c in consultations_db.values() 
        if c["doctor_id"] == username
    ]
    return user_consultations

# WebSocket for live transcription
@app.websocket("/ws/transcribe/{consultation_id}")
async def websocket_transcribe(websocket: WebSocket, consultation_id: str):
    await websocket.accept()
    
    consultation = consultations_db.get(consultation_id)
    if not consultation:
        await websocket.close(code=1008)
        return
    
    try:
        while True:
            # Receive audio data from frontend
            data = await websocket.receive_bytes()
            
            # Process with AssemblyAI streaming
            # For simplicity, we'll simulate speaker detection
            # In production, use AssemblyAI's real-time transcription with speaker labels
            
            # Send back mock transcription (replace with actual AssemblyAI streaming)
            response = {
                "type": "transcript",
                "text": "Sample transcription text",
                "speaker": "Doctor",  # or "Patient"
                "timestamp": datetime.utcnow().isoformat(),
                "confidence": 0.95,
                "is_final": False
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for consultation {consultation_id}")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close(code=1011)

# Helper function for summary generation
def generate_clinical_summary(transcript: str) -> dict:
    """
    Generate structured clinical summary from transcript.
    In production, use AssemblyAI LeMUR or OpenAI GPT-4.
    """
    # This is a simplified version - replace with actual LLM call
    return {
        "identifiers": "Name: Patient Name\nAge: [From transcript]\nSex: [From transcript]\nLocation: [From transcript]",
        "history": "Chief Complaint: Extracted from transcript\nHistory of Present Illness: Key details from patient's description",
        "examination": "Vital signs and physical examination findings mentioned in the consultation",
        "diagnoses": "Primary and secondary diagnoses based on the consultation",
        "treatment": "Prescribed medications and treatment plan",
        "advice": "Lifestyle modifications and patient counseling provided",
        "next_steps": "Follow-up appointments and investigations recommended"
    }

def extract_section(transcript: str, section: str) -> str:
    """Extract relevant information for each section from transcript"""
    # Simplified extraction logic
    return f"Information extracted from transcript related to {section}. (Replace with actual NLP extraction)"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)