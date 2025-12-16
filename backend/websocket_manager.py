# backend/websocket_manager.py
from fastapi import WebSocket
from typing import Dict, List
import asyncio
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time transcription"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.consultation_transcripts: Dict[str, List[dict]] = {}
        
    async def connect(self, consultation_id: str, websocket: WebSocket):
        """Accept and store a new WebSocket connection"""
        await websocket.accept()
        self.active_connections[consultation_id] = websocket
        self.consultation_transcripts[consultation_id] = []
        logger.info(f"WebSocket connected for consultation {consultation_id}")
        
    def disconnect(self, consultation_id: str):
        """Remove a WebSocket connection"""
        if consultation_id in self.active_connections:
            del self.active_connections[consultation_id]
        logger.info(f"WebSocket disconnected for consultation manager {consultation_id}")
        
    async def send_transcript(self, consultation_id: str, transcript_data: dict):
        """Send transcript data to the connected client"""
        if consultation_id in self.active_connections:
            websocket = self.active_connections[consultation_id]
            try:
                await websocket.send_json(transcript_data)
                
                # Store in consultation transcript if final
                if transcript_data.get("is_final", False):
                    self.consultation_transcripts[consultation_id].append({
                        "text": transcript_data["text"],
                        "speaker": transcript_data["speaker"],
                        "timestamp": transcript_data.get("timestamp", datetime.utcnow().isoformat()),
                        "confidence": transcript_data.get("confidence", 0.95)
                    })
                    
            except Exception as e:
                logger.error(f"Error sending transcript: {e}")
                self.disconnect(consultation_id)
                
    async def broadcast_status(self, consultation_id: str, status: str, message: str = ""):
        """Send status update to the connected client"""
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
        """Retrieve stored transcript for a consultation"""
        return self.consultation_transcripts.get(consultation_id, [])
    
    def clear_transcript(self, consultation_id: str):
        """Clear transcript data for a consultation"""
        if consultation_id in self.consultation_transcripts:
            del self.consultation_transcripts[consultation_id]

# Global connection manager instance
manager = ConnectionManager()