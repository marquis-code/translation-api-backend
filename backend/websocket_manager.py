from fastapi import WebSocket
from typing import Dict, List, Optional
import asyncio
import json
from datetime import datetime
import logging
import time
import hashlib

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections with smart duplicate prevention"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.consultation_transcripts: Dict[str, List[dict]] = {}
        self.consultation_languages: Dict[str, str] = {}
        # Simplified duplicate tracking - just track last few transcripts
        self.recent_sends: Dict[str, List[tuple]] = {}  # consultation_id -> [(text, speaker, time)]
        
    async def connect(self, consultation_id: str, websocket: WebSocket, language: str = "en"):
        """Accept and store a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[consultation_id] = websocket
        self.consultation_transcripts[consultation_id] = []
        self.consultation_languages[consultation_id] = language
        self.recent_sends[consultation_id] = []
        logger.info(f"✅ WebSocket connected: {consultation_id} (lang: {language})")
        
    def disconnect(self, consultation_id: str):
        """Remove a WebSocket connection"""
        if consultation_id in self.active_connections:
            del self.active_connections[consultation_id]
        if consultation_id in self.consultation_languages:
            del self.consultation_languages[consultation_id]
        if consultation_id in self.recent_sends:
            del self.recent_sends[consultation_id]
        logger.info(f"🔌 WebSocket disconnected: {consultation_id}")
        
    def set_language(self, consultation_id: str, language: str):
        """Update the language for a consultation"""
        self.consultation_languages[consultation_id] = language
        
    def get_language(self, consultation_id: str) -> str:
        """Get the language for a consultation"""
        return self.consultation_languages.get(consultation_id, "en")
    
    def _is_duplicate(self, consultation_id: str, text: str, speaker: str) -> bool:
        """
        Smart duplicate detection - only block true duplicates
        Returns True if duplicate, False if unique
        """
        if consultation_id not in self.recent_sends:
            return False
        
        recent = self.recent_sends[consultation_id]
        text_lower = text.lower().strip()
        current_time = time.time()
        
        # Check last 5 sends
        for recent_text, recent_speaker, recent_time in recent[-5:]:
            # Exact match from same speaker within 2 seconds
            if recent_speaker == speaker and recent_text == text_lower:
                time_diff = current_time - recent_time
                if time_diff < 2.0:
                    logger.warning(f"🚫 WS Duplicate: {text[:30]} ({time_diff:.1f}s ago)")
                    return True
        
        return False
        
    async def send_transcript(self, consultation_id: str, transcript_data: dict):
        """Send transcript with SMART duplicate prevention"""
        if consultation_id not in self.active_connections:
            logger.warning(f"⚠️ No active connection for {consultation_id}")
            return
        
        websocket = self.active_connections[consultation_id]
        
        try:
            text = transcript_data['text'].strip()
            speaker = transcript_data['speaker']
            
            # CHECK 1: Empty or very short text
            if len(text) < 2:
                logger.warning(f"🚫 Text too short: '{text}'")
                return
            
            # CHECK 2: Duplicate check
            if self._is_duplicate(consultation_id, text, speaker):
                return
            
            # All checks passed - send it!
            await websocket.send_json(transcript_data)
            logger.info(f"✅ WebSocket SENT: [{speaker}] {text[:60]}")
            
            # Track this send
            current_time = time.time()
            if consultation_id not in self.recent_sends:
                self.recent_sends[consultation_id] = []
            
            self.recent_sends[consultation_id].append((text.lower().strip(), speaker, current_time))
            
            # Keep only last 10 sends
            if len(self.recent_sends[consultation_id]) > 10:
                self.recent_sends[consultation_id].pop(0)
            
            # Store final transcripts
            if transcript_data.get("is_final", False):
                self.consultation_transcripts[consultation_id].append({
                    "text": text,
                    "original_text": transcript_data.get("original_text", text),
                    "speaker": speaker,
                    "timestamp": transcript_data.get("timestamp", datetime.utcnow().isoformat()),
                    "confidence": transcript_data.get("confidence", 0.95),
                    "language": transcript_data.get("language", "en")
                })
                logger.info(f"💾 Stored: [{speaker}] {text[:40]}")
                    
        except Exception as e:
            logger.error(f"❌ Error sending transcript: {e}")
            self.disconnect(consultation_id)
                
    async def broadcast_status(self, consultation_id: str, status: str, message: str = ""):
        """Send status update"""
        if consultation_id in self.active_connections:
            websocket = self.active_connections[consultation_id]
            try:
                await websocket.send_json({
                    "type": "status",
                    "status": status,
                    "message": message,
                    "timestamp": datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"Error sending status: {e}")
                
    def get_transcript(self, consultation_id: str) -> List[dict]:
        """Retrieve stored transcript"""
        return self.consultation_transcripts.get(consultation_id, [])
    
    def clear_transcript(self, consultation_id: str):
        """Clear transcript data"""
        if consultation_id in self.consultation_transcripts:
            del self.consultation_transcripts[consultation_id]
        if consultation_id in self.consultation_languages:
            del self.consultation_languages[consultation_id]
        if consultation_id in self.recent_sends:
            del self.recent_sends[consultation_id]

manager = ConnectionManager()