
# backend/streaming_service.py
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    TerminationEvent,
    TurnEvent,
)
import asyncio
from typing import Callable, Optional, Iterator
import logging
from datetime import datetime
import json
import threading
import queue
from dotenv import load_dotenv
from os import getenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
# HARDCODED API KEY
# HARDCODED_API_KEY = "7f9a4eb77c1d4fc1bfb13561a7ef3f14"
HARDCODED_API_KEY = getenv("ASSEMBLYAI_API_KEY")
class AudioStreamSource:
    """Custom audio source that reads from a queue"""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_active = True
        
    def __iter__(self) -> Iterator[bytes]:
        """Yield audio chunks from the queue"""
        while self.is_active:
            try:
                # Get audio data with timeout to allow checking is_active
                audio_chunk = self.audio_queue.get(timeout=0.1)
                if audio_chunk is not None:
                    yield audio_chunk
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in audio stream: {e}")
                break
    
    def add_audio(self, audio_data: bytes):
        """Add audio data to the queue"""
        if self.is_active:
            self.audio_queue.put(audio_data)
    
    def stop(self):
        """Stop the audio stream"""
        self.is_active = False
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break


class TranscriptionService:
    """Real-time transcription service using AssemblyAI Streaming API v3"""
    
    def __init__(self, api_key: str, consultation_id: str):
        # FORCE USE HARDCODED KEY
        self.api_key = HARDCODED_API_KEY
        
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("API key is empty!")
        
        self.consultation_id = consultation_id
        self.client = None
        self.audio_source = None
        self.streaming_thread = None
        self.is_running = False
        self.current_speaker = "Doctor"
        self.on_transcript = None
        self.session_id = None
        
        logger.info(f"🔑 TranscriptionService initialized with API key: {self.api_key[:15]}...")
        
    def start(self, on_transcript_callback: Callable):
        """Start the streaming transcription service"""
        self.is_running = True
        self.on_transcript = on_transcript_callback
        
        try:
            logger.info(f"🚀 Creating StreamingClient with key: {self.api_key[:15]}...")
            
            # Create StreamingClient with v3 API
            self.client = StreamingClient(
                StreamingClientOptions(
                    api_key=self.api_key,
                    api_host="streaming.assemblyai.com",
                )
            )
            
            # Register event handlers
            self.client.on(StreamingEvents.Begin, self._on_begin)
            self.client.on(StreamingEvents.Turn, self._on_turn)
            self.client.on(StreamingEvents.Termination, self._on_terminated)
            self.client.on(StreamingEvents.Error, self._on_error)
            
            # Connect to streaming service
            logger.info("🔌 Connecting to AssemblyAI streaming...")
            self.client.connect(
                StreamingParameters(
                    sample_rate=16000,
                    format_turns=True,
                )
            )
            
            # Create audio source
            self.audio_source = AudioStreamSource()
            
            # Start streaming in a separate thread
            self.streaming_thread = threading.Thread(
                target=self._stream_audio,
                daemon=True
            )
            self.streaming_thread.start()
            
            logger.info(f"✅ Transcription service started for consultation {self.consultation_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start transcription service: {e}", exc_info=True)
            self.is_running = False
            raise
    
    def _stream_audio(self):
        """Stream audio from the audio source to AssemblyAI"""
        try:
            logger.info("🎤 Starting audio streaming thread...")
            self.client.stream(self.audio_source)
            logger.info("🎤 Audio streaming thread completed")
        except Exception as e:
            logger.error(f"❌ Error in audio streaming thread: {e}", exc_info=True)
            self.is_running = False
    
    def _on_begin(self, client: StreamingClient, event: BeginEvent):
        """Handle session begin event"""
        self.session_id = event.id
        logger.info(f"✅ Session started: {event.id}")
    
    def _on_turn(self, client: StreamingClient, event: TurnEvent):
        """Handle turn event (transcription data)"""
        if not event.transcript:
            return
        
        # Detect speaker
        speaker = self._detect_speaker(event.transcript)
        
        # Prepare transcript data
        transcript_data = {
            "type": "transcript",
            "text": event.transcript,
            "speaker": speaker,
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": 0.95,  # v3 API doesn't provide confidence per turn
            "is_final": event.end_of_turn,
        }
        
        # Call the callback
        if self.on_transcript:
            try:
                self.on_transcript(transcript_data)
            except Exception as e:
                logger.error(f"❌ Error in transcript callback: {e}")
        
        if event.end_of_turn:
            logger.info(f"📝 Turn complete: {event.transcript[:50]}... (Speaker: {speaker})")
        else:
            logger.debug(f"📝 Partial: {event.transcript[:30]}...")
    
    def _on_terminated(self, client: StreamingClient, event: TerminationEvent):
        """Handle session termination"""
        logger.info(f"🔌 Session terminated: {event.audio_duration_seconds}s of audio processed")
        self.is_running = False
    
    def _on_error(self, client: StreamingClient, error: StreamingError):
        """Handle streaming errors"""
        logger.error(f"❌ STREAMING ERROR: {error}")
        logger.error(f"Error type: {type(error)}")
        
        # Check for auth errors
        error_str = str(error).lower()
        if "not authorized" in error_str or "unauthorized" in error_str or "401" in error_str:
            logger.error("=" * 80)
            logger.error("🔐 AUTHORIZATION ERROR DETECTED!")
            logger.error(f"API Key being used: {self.api_key[:15]}...")
            logger.error(f"API Key length: {len(self.api_key)}")
            logger.error("=" * 80)
            logger.error("SOLUTIONS:")
            logger.error("1. Get a NEW API key from: https://www.assemblyai.com/app/account")
            logger.error("2. Make sure you're on a paid plan (free tier may not support streaming)")
            logger.error("3. Check if your account has available credits")
            logger.error("=" * 80)
        
        self.is_running = False
    
    def _detect_speaker(self, text: str) -> str:
        """Detect speaker based on medical terminology"""
        doctor_keywords = [
            "prescribe", "diagnosis", "diagnose", "examine", "examination",
            "medication", "treatment", "let me check", "blood pressure",
            "temperature", "symptoms suggest", "vital signs", "assessment",
            "impression", "plan", "recommend", "advise", "follow up"
        ]
        
        patient_keywords = [
            "i feel", "i have", "my pain", "i'm experiencing", "it hurts",
            "since last", "i noticed", "i've been", "i am", "i was",
            "my symptoms", "i can't", "i cannot", "bothering me"
        ]
        
        text_lower = text.lower()
        
        doctor_score = sum(1 for keyword in doctor_keywords if keyword in text_lower)
        patient_score = sum(1 for keyword in patient_keywords if keyword in text_lower)
        
        if doctor_score > patient_score:
            self.current_speaker = "Doctor"
        elif patient_score > doctor_score:
            self.current_speaker = "Patient"
        
        return self.current_speaker
    
    def send_audio(self, audio_data: bytes):
        """Send audio data to the streaming service"""
        if self.audio_source and self.is_running:
            try:
                self.audio_source.add_audio(audio_data)
            except Exception as e:
                logger.error(f"❌ Error sending audio: {e}")
                self.is_running = False
        else:
            logger.warning(f"⚠️ Cannot send audio - service running: {self.is_running}")
    
    def stop(self):
        """Stop the transcription service"""
        logger.info("🛑 Stopping transcription service...")
        self.is_running = False
        
        if self.audio_source:
            self.audio_source.stop()
        
        if self.client:
            try:
                logger.info("🛑 Disconnecting from streaming service...")
                self.client.disconnect(terminate=True)
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
        
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=2.0)
        
        logger.info(f"🛑 Transcription service stopped for {self.consultation_id}")


class SummaryService:
    """Service for generating clinical summaries"""
    
    def __init__(self, api_key: str):
        self.api_key = HARDCODED_API_KEY
        aai.settings.api_key = self.api_key
        logger.info(f"📋 SummaryService using API key: {self.api_key[:15]}...")
    
    async def generate_summary(self, transcript: str, segments: list) -> dict:
        """Generate structured clinical summary from transcript segments"""
        formatted_transcript = "\n".join([
            f"{seg['speaker']}: {seg['text']}" for seg in segments
        ])
        
        try:
            summary = self._extract_structured_summary(formatted_transcript, segments)
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise
    
    def _extract_structured_summary(self, transcript: str, segments: list) -> dict:
        """Extract structured clinical information from transcript"""
        
        identifiers = self._extract_identifiers(segments)
        history = self._extract_history(segments)
        examination = self._extract_examination(segments)
        diagnoses = self._extract_diagnoses(segments)
        treatment = self._extract_treatment(segments)
        advice = self._extract_advice(segments)
        next_steps = self._extract_next_steps(segments)
        
        return {
            "identifiers": identifiers,
            "history": history,
            "examination": examination,
            "diagnoses": diagnoses,
            "treatment": treatment,
            "advice": advice,
            "next_steps": next_steps
        }
    
    def _extract_identifiers(self, segments: list) -> str:
        """Extract patient identifiers from initial segments"""
        identifiers = {"name": None, "age": None, "sex": None, "location": None}
        
        for seg in segments[:10]:
            text_lower = seg['text'].lower()
            
            if any(phrase in text_lower for phrase in ["my name is", "i'm", "i am"]):
                words = seg['text'].split()
                for i, word in enumerate(words):
                    if word.lower() in ["name", "i'm", "am"] and i + 1 < len(words):
                        identifiers["name"] = " ".join(words[i+1:min(i+3, len(words))])
                        break
            
            if "year" in text_lower or "age" in text_lower:
                import re
                age_match = re.search(r'\b(\d{1,3})\s*(?:year|yr)', text_lower)
                if age_match:
                    identifiers["age"] = age_match.group(1)
            
            if "male" in text_lower or "female" in text_lower:
                identifiers["sex"] = "Male" if "male" in text_lower and "female" not in text_lower else "Female"
        
        result = f"""Name: {identifiers['name'] or '[To be filled]'}
Age: {identifiers['age'] or '[To be filled]'} years
Sex: {identifiers['sex'] or '[To be filled]'}
Location: {identifiers['location'] or '[To be filled]'}"""
        
        return result
    
    def _extract_history(self, segments: list) -> str:
        """Extract medical history"""
        history_keywords = [
            "complain", "complaint", "pain", "symptom", "problem", "issue",
            "feel", "feeling", "since", "ago", "started", "began", "history"
        ]
        
        relevant_segments = [
            f"- {seg['text']}" for seg in segments 
            if any(keyword in seg['text'].lower() for keyword in history_keywords)
            and seg['speaker'] in ['Patient', 'Doctor']
        ]
        
        if relevant_segments:
            chief_complaint = "\n".join(relevant_segments[:8])
            return f"Chief Complaint and History:\n{chief_complaint}"
        
        return "Chief Complaint: [To be documented from consultation]"
    
    def _extract_examination(self, segments: list) -> str:
        """Extract examination findings"""
        exam_keywords = [
            "blood pressure", "bp", "temperature", "temp", "heart rate",
            "pulse", "examine", "examination", "check", "checked",
            "vital", "respiratory", "breath"
        ]
        
        relevant_segments = [
            f"- {seg['text']}" for seg in segments 
            if any(keyword in seg['text'].lower() for keyword in exam_keywords)
            and seg['speaker'] == 'Doctor'
        ]
        
        if relevant_segments:
            findings = "\n".join(relevant_segments[:8])
            return f"Examination Findings:\n{findings}"
        
        return "Examination Findings: [Physical examination to be documented]"
    
    def _extract_diagnoses(self, segments: list) -> str:
        """Extract diagnoses and assessment"""
        diagnosis_keywords = [
            "diagnos", "appears to be", "likely", "probably", "condition",
            "disease", "suffering from", "assessment", "impression"
        ]
        
        relevant_segments = [
            f"- {seg['text']}" for seg in segments 
            if any(keyword in seg['text'].lower() for keyword in diagnosis_keywords)
            and seg['speaker'] == 'Doctor'
        ]
        
        if relevant_segments:
            assessment = "\n".join(relevant_segments[:6])
            return f"Clinical Assessment:\n{assessment}"
        
        return "Clinical Assessment: [Diagnosis to be determined]"
    
    def _extract_treatment(self, segments: list) -> str:
        """Extract treatment plan"""
        treatment_keywords = [
            "prescribe", "prescription", "medication", "medicine", "tablet",
            "capsule", "treatment", "dose", "dosage", "take", "mg", "ml"
        ]
        
        relevant_segments = [
            f"- {seg['text']}" for seg in segments 
            if any(keyword in seg['text'].lower() for keyword in treatment_keywords)
            and seg['speaker'] == 'Doctor'
        ]
        
        if relevant_segments:
            plan = "\n".join(relevant_segments[:8])
            return f"Treatment Plan:\n{plan}"
        
        return "Treatment Plan: [Medications and interventions to be prescribed]"
    
    def _extract_advice(self, segments: list) -> str:
        """Extract advice and counseling"""
        advice_keywords = [
            "advise", "advice", "recommend", "should", "shouldn't",
            "avoid", "diet", "exercise", "lifestyle", "rest", "drink",
            "eat", "sleep"
        ]
        
        relevant_segments = [
            f"- {seg['text']}" for seg in segments 
            if any(keyword in seg['text'].lower() for keyword in advice_keywords)
            and seg['speaker'] == 'Doctor'
        ]
        
        if relevant_segments:
            counseling = "\n".join(relevant_segments[:8])
            return f"Patient Counseling:\n{counseling}"
        
        return "Patient Counseling: [Lifestyle modifications and advice to be discussed]"
    
    def _extract_next_steps(self, segments: list) -> str:
        """Extract follow-up plans"""
        followup_keywords = [
            "follow", "followup", "follow-up", "next", "appointment",
            "come back", "return", "visit", "test", "investigation",
            "lab", "scan", "x-ray"
        ]
        
        relevant_segments = [
            f"- {seg['text']}" for seg in segments 
            if any(keyword in seg['text'].lower() for keyword in followup_keywords)
            and seg['speaker'] == 'Doctor'
        ]
        
        if relevant_segments:
            followup = "\n".join(relevant_segments[:6])
            return f"Follow-up Plan:\n{followup}"
        
        return "Follow-up: Schedule follow-up appointment in 1-2 weeks for review"