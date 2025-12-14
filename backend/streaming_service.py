# backend/streaming_service.py
import assemblyai as aai
from assemblyai.streaming.v3 import (
    BeginEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingParameters,
    StreamingSessionParameters,
    TerminationEvent,
    TurnEvent,
)
import asyncio
from typing import Callable, Optional
import logging
from queue import Queue
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionService:
    def __init__(self, api_key: str, consultation_id: str):
        self.api_key = api_key
        self.consultation_id = consultation_id
        self.client: Optional[StreamingClient] = None
        self.transcript_queue = Queue()
        self.is_running = False
        self.current_speaker = "Doctor"  # Default speaker
        
    def start(self, on_transcript_callback: Callable):
        """Start the streaming transcription service"""
        self.is_running = True
        self.on_transcript = on_transcript_callback
        
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
        
        # Connect with multilingual support
        self.client.connect(
            StreamingParameters(
                sample_rate=16000,
                format_turns=True,
                # Enable speaker diarization (if available in your plan)
                # speaker_labels=True,
            )
        )
        
        logger.info(f"Transcription service started for consultation {self.consultation_id}")
        
    def _on_begin(self, client, event: BeginEvent):
        logger.info(f"Session started: {event.id}")
        
    def _on_turn(self, client, event: TurnEvent):
        """Handle transcription turns"""
        transcript_data = {
            "text": event.transcript,
            "speaker": self._detect_speaker(event.transcript),
            "timestamp": event.created_at if hasattr(event, 'created_at') else None,
            "confidence": getattr(event, 'confidence', 0.95),
            "is_final": event.turn_is_formatted,
            "end_of_turn": event.end_of_turn
        }
        
        # Call the callback with transcript data
        if self.on_transcript:
            self.on_transcript(transcript_data)
        
        logger.info(f"Turn: {event.transcript[:50]}... (Speaker: {transcript_data['speaker']})")
        
        # Request formatted turns if end of turn reached
        if event.end_of_turn and not event.turn_is_formatted:
            params = StreamingSessionParameters(format_turns=True)
            client.set_params(params)
    
    def _on_terminated(self, client, event: TerminationEvent):
        logger.info(f"Session terminated: {event.audio_duration_seconds}s of audio processed")
        self.is_running = False
        
    def _on_error(self, client, error: StreamingError):
        logger.error(f"Streaming error: {error}")
        self.is_running = False
    
    def _detect_speaker(self, text: str) -> str:
        """
        Simple speaker detection based on medical terminology.
        In production, use proper speaker diarization.
        """
        # Keywords that suggest doctor is speaking
        doctor_keywords = [
            "prescribe", "diagnosis", "examine", "medication", "treatment",
            "let me check", "blood pressure", "temperature", "symptoms suggest"
        ]
        
        # Keywords that suggest patient is speaking
        patient_keywords = [
            "i feel", "i have", "my pain", "i'm experiencing", "it hurts",
            "since last", "i noticed", "i've been"
        ]
        
        text_lower = text.lower()
        
        doctor_score = sum(1 for keyword in doctor_keywords if keyword in text_lower)
        patient_score = sum(1 for keyword in patient_keywords if keyword in text_lower)
        
        if doctor_score > patient_score:
            self.current_speaker = "Doctor"
        elif patient_score > doctor_score:
            self.current_speaker = "Patient"
        # else keep current speaker
        
        return self.current_speaker
    
    def send_audio(self, audio_data: bytes):
        """Send audio data to the streaming service"""
        if self.client and self.is_running:
            try:
                self.client.stream_audio(audio_data)
            except Exception as e:
                logger.error(f"Error sending audio: {e}")
    
    def stop(self):
        """Stop the transcription service"""
        if self.client:
            self.client.disconnect(terminate=True)
            self.is_running = False
            logger.info("Transcription service stopped")


class BatchTranscriptionService:
    """Service for batch transcription with speaker diarization"""
    
    def __init__(self, api_key: str):
        aai.settings.api_key = api_key
        
    async def transcribe_audio(self, audio_url: str, languages: list = None) -> dict:
        """
        Transcribe audio file with speaker diarization and multilingual support
        """
        if languages is None:
            languages = ["en", "hi"]  # English and Hindi
        
        config = aai.TranscriptionConfig(
            speaker_labels=True,  # Enable speaker diarization
            language_detection=True,  # Auto-detect language
            multichannel=False,
            speech_model="best",  # Use best available model
        )
        
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(audio_url)
        
        if transcript.status == "error":
            raise RuntimeError(f"Transcription failed: {transcript.error}")
        
        # Extract speaker-labeled segments
        segments = []
        for utterance in transcript.utterances:
            segments.append({
                "text": utterance.text,
                "speaker": utterance.speaker,
                "start": utterance.start,
                "end": utterance.end,
                "confidence": utterance.confidence
            })
        
        return {
            "transcript": transcript.text,
            "segments": segments,
            "language": getattr(transcript, 'language_code', 'en')
        }


class SummaryService:
    """Service for generating clinical summaries using LLM"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    async def generate_summary(self, transcript: str, segments: list) -> dict:
        """
        Generate structured clinical summary from transcript.
        Uses AssemblyAI LeMUR or fallback to rule-based extraction.
        """
        
        # Format transcript with speakers
        formatted_transcript = "\n".join([
            f"{seg['speaker']}: {seg['text']}" for seg in segments
        ])
        
        try:
            # Use AssemblyAI LeMUR for advanced summarization
            # Note: LeMUR requires a specific plan
            summary_prompt = f"""
            Analyze the following medical consultation transcript and extract structured information:
            
            {formatted_transcript}
            
            Provide a JSON response with the following sections:
            1. identifiers: Patient name, age, sex, and location
            2. history: Chief complaints, history of present illness, past medical history
            3. examination: Physical examination findings
            4. diagnoses: Clinical assessment and diagnoses
            5. treatment: Medications and treatment plan
            6. advice: Lifestyle changes and patient education
            7. next_steps: Follow-up appointments and investigations
            """
            
            # For demo, use rule-based extraction
            summary = self._extract_structured_summary(formatted_transcript)
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback to basic extraction
            return self._extract_structured_summary(formatted_transcript)
    
    def _extract_structured_summary(self, transcript: str) -> dict:
        """Rule-based extraction of clinical information"""
        lines = transcript.split('\n')
        
        identifiers = self._extract_identifiers(lines)
        history = self._extract_history(lines)
        examination = self._extract_examination(lines)
        diagnoses = self._extract_diagnoses(lines)
        treatment = self._extract_treatment(lines)
        advice = self._extract_advice(lines)
        next_steps = self._extract_next_steps(lines)
        
        return {
            "identifiers": identifiers,
            "history": history,
            "examination": examination,
            "diagnoses": diagnoses,
            "treatment": treatment,
            "advice": advice,
            "next_steps": next_steps
        }
    
    def _extract_identifiers(self, lines: list) -> str:
        """Extract patient identifiers"""
        identifiers = {
            "name": None,
            "age": None,
            "sex": None,
            "location": None
        }
        
        for line in lines[:10]:  # Check first 10 lines
            line_lower = line.lower()
            if "name" in line_lower or "i'm" in line_lower or "i am" in line_lower:
                identifiers["name"] = line.split(":", 1)[-1].strip() if ":" in line else "[From transcript]"
            if "age" in line_lower or "year" in line_lower:
                identifiers["age"] = "[From transcript]"
            if "male" in line_lower or "female" in line_lower:
                identifiers["sex"] = "Male" if "male" in line_lower and "female" not in line_lower else "Female"
        
        result = f"""Name: {identifiers['name'] or '[To be filled]'}
Age: {identifiers['age'] or '[To be filled]'}
Sex: {identifiers['sex'] or '[To be filled]'}
Location: {identifiers['location'] or '[To be filled]'}"""
        
        return result
    
    def _extract_history(self, lines: list) -> str:
        """Extract medical history"""
        history_keywords = ["complain", "pain", "symptom", "problem", "issue", "feel", "since", "ago"]
        relevant_lines = [
            line for line in lines 
            if any(keyword in line.lower() for keyword in history_keywords)
        ]
        
        if relevant_lines:
            return "Chief Complaint:\n" + "\n".join(relevant_lines[:5])
        return "Chief Complaint: [To be extracted from consultation]"
    
    def _extract_examination(self, lines: list) -> str:
        """Extract examination findings"""
        exam_keywords = ["blood pressure", "temperature", "heart rate", "examine", "check", "pulse"]
        relevant_lines = [
            line for line in lines 
            if any(keyword in line.lower() for keyword in exam_keywords)
        ]
        
        if relevant_lines:
            return "Examination Findings:\n" + "\n".join(relevant_lines[:5])
        return "Examination Findings: [To be documented]"
    
    def _extract_diagnoses(self, lines: list) -> str:
        """Extract diagnoses"""
        diagnosis_keywords = ["diagnos", "appears to be", "likely", "condition", "disease"]
        relevant_lines = [
            line for line in lines 
            if any(keyword in line.lower() for keyword in diagnosis_keywords)
        ]
        
        if relevant_lines:
            return "Assessment:\n" + "\n".join(relevant_lines[:5])
        return "Assessment: [To be determined]"
    
    def _extract_treatment(self, lines: list) -> str:
        """Extract treatment plan"""
        treatment_keywords = ["prescribe", "medication", "tablet", "treatment", "dose", "medicine"]
        relevant_lines = [
            line for line in lines 
            if any(keyword in line.lower() for keyword in treatment_keywords)
        ]
        
        if relevant_lines:
            return "Treatment Plan:\n" + "\n".join(relevant_lines[:5])
        return "Treatment Plan: [To be prescribed]"
    
    def _extract_advice(self, lines: list) -> str:
        """Extract advice and counseling"""
        advice_keywords = ["advise", "recommend", "should", "avoid", "diet", "exercise", "lifestyle"]
        relevant_lines = [
            line for line in lines 
            if any(keyword in line.lower() for keyword in advice_keywords)
        ]
        
        if relevant_lines:
            return "Patient Counseling:\n" + "\n".join(relevant_lines[:5])
        return "Patient Counseling: [Lifestyle modifications to be discussed]"
    
    def _extract_next_steps(self, lines: list) -> str:
        """Extract follow-up plans"""
        followup_keywords = ["follow", "next", "appointment", "come back", "return", "test", "investigation"]
        relevant_lines = [
            line for line in lines 
            if any(keyword in line.lower() for keyword in followup_keywords)
        ]
        
        if relevant_lines:
            return "Follow-up:\n" + "\n".join(relevant_lines[:5])
        return "Follow-up: Schedule appointment in 2 weeks for review"