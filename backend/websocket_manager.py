
# # from fastapi import WebSocket
# # from typing import Dict, List, Optional
# # import asyncio
# # import json
# # from datetime import datetime
# # import logging

# # logger = logging.getLogger(__name__)

# # class ConnectionManager:
# #     """Manages WebSocket connections for real-time transcription"""
    
# #     def __init__(self):
# #         self.active_connections: Dict[str, WebSocket] = {}
# #         self.consultation_transcripts: Dict[str, List[dict]] = {}
# #         self.consultation_languages: Dict[str, str] = {}  # Track language per consultation
        
# #     async def connect(self, consultation_id: str, websocket: WebSocket, language: str = "en"):
# #         """Accept and store a new WebSocket connection"""
# #         await websocket.accept()
# #         self.active_connections[consultation_id] = websocket
# #         self.consultation_transcripts[consultation_id] = []
# #         self.consultation_languages[consultation_id] = language
# #         logger.info(f"WebSocket connected for consultation {consultation_id} with language {language}")
        
# #     def disconnect(self, consultation_id: str):
# #         """Remove a WebSocket connection"""
# #         if consultation_id in self.active_connections:
# #             del self.active_connections[consultation_id]
# #         if consultation_id in self.consultation_languages:
# #             del self.consultation_languages[consultation_id]
# #         logger.info(f"WebSocket disconnected for consultation {consultation_id}")
        
# #     def set_language(self, consultation_id: str, language: str):
# #         """Update the language for a consultation"""
# #         self.consultation_languages[consultation_id] = language
# #         logger.info(f"Language set to {language} for consultation {consultation_id}")
        
# #     def get_language(self, consultation_id: str) -> str:
# #         """Get the language for a consultation"""
# #         return self.consultation_languages.get(consultation_id, "en")
        
# #     async def send_transcript(self, consultation_id: str, transcript_data: dict):
# #         """Send transcript data to the connected client"""
# #         if consultation_id in self.active_connections:
# #             websocket = self.active_connections[consultation_id]
# #             try:
# #                 await websocket.send_json(transcript_data)
                
# #                 # Store both original and translated text if final
# #                 if transcript_data.get("is_final", False):
# #                     self.consultation_transcripts[consultation_id].append({
# #                         "text": transcript_data["text"],
# #                         "original_text": transcript_data.get("original_text", transcript_data["text"]),
# #                         "speaker": transcript_data["speaker"],
# #                         "timestamp": transcript_data.get("timestamp", datetime.utcnow().isoformat()),
# #                         "confidence": transcript_data.get("confidence", 0.95),
# #                         "language": transcript_data.get("language", "en")
# #                     })
                    
# #             except Exception as e:
# #                 logger.error(f"Error sending transcript: {e}")
# #                 self.disconnect(consultation_id)
                
# #     async def broadcast_status(self, consultation_id: str, status: str, message: str = ""):
# #         """Send status update to the connected client"""
# #         if consultation_id in self.active_connections:
# #             websocket = self.active_connections[consultation_id]
# #             try:
# #                 await websocket.send_json({
# #                     "type": "status",
# #                     "status": status,
# #                     "message": message,
# #                     "timestamp": datetime.utcnow().isoformat()
# #                 })
# #             except Exception as e:
# #                 logger.error(f"Error sending status: {e}")
                
# #     def get_transcript(self, consultation_id: str) -> List[dict]:
# #         """Retrieve stored transcript for a consultation"""
# #         return self.consultation_transcripts.get(consultation_id, [])
    
# #     def clear_transcript(self, consultation_id: str):
# #         """Clear transcript data for a consultation"""
# #         if consultation_id in self.consultation_transcripts:
# #             del self.consultation_transcripts[consultation_id]
# #         if consultation_id in self.consultation_languages:
# #             del self.consultation_languages[consultation_id]

# # manager = ConnectionManager()


# from fastapi import WebSocket
# from typing import Dict, List, Optional
# import asyncio
# import json
# from datetime import datetime
# import logging

# logger = logging.getLogger(__name__)

# class ConnectionManager:
#     """Manages WebSocket connections for real-time transcription with deduplication"""
    
#     def __init__(self):
#         self.active_connections: Dict[str, WebSocket] = {}
#         self.consultation_transcripts: Dict[str, List[dict]] = {}
#         self.consultation_languages: Dict[str, str] = {}
#         self.last_transcript_hash: Dict[str, str] = {}  # Track last sent transcript per consultation
        
#     async def connect(self, consultation_id: str, websocket: WebSocket, language: str = "en"):
#         """Accept and store a new WebSocket connection"""
#         await websocket.accept()
#         self.active_connections[consultation_id] = websocket
#         self.consultation_transcripts[consultation_id] = []
#         self.consultation_languages[consultation_id] = language
#         self.last_transcript_hash[consultation_id] = ""
#         logger.info(f"WebSocket connected for consultation {consultation_id} with language {language}")
        
#     def disconnect(self, consultation_id: str):
#         """Remove a WebSocket connection"""
#         if consultation_id in self.active_connections:
#             del self.active_connections[consultation_id]
#         if consultation_id in self.consultation_languages:
#             del self.consultation_languages[consultation_id]
#         if consultation_id in self.last_transcript_hash:
#             del self.last_transcript_hash[consultation_id]
#         logger.info(f"WebSocket disconnected for consultation {consultation_id}")
        
#     def set_language(self, consultation_id: str, language: str):
#         """Update the language for a consultation"""
#         self.consultation_languages[consultation_id] = language
#         logger.info(f"Language set to {language} for consultation {consultation_id}")
        
#     def get_language(self, consultation_id: str) -> str:
#         """Get the language for a consultation"""
#         return self.consultation_languages.get(consultation_id, "en")
    
#     def _create_transcript_hash(self, transcript_data: dict) -> str:
#         """Create a unique hash for transcript to detect duplicates"""
#         return f"{transcript_data['text'].strip()}:{transcript_data['speaker']}:{transcript_data.get('timestamp', '')[:16]}"
        
#     async def send_transcript(self, consultation_id: str, transcript_data: dict):
#         """Send transcript data to the connected client with deduplication"""
#         if consultation_id in self.active_connections:
#             websocket = self.active_connections[consultation_id]
            
#             try:
#                 # CRITICAL: Check for duplicates before sending
#                 transcript_hash = self._create_transcript_hash(transcript_data)
                
#                 # Skip if this is a duplicate
#                 if transcript_hash == self.last_transcript_hash.get(consultation_id, ""):
#                     logger.warning(f"⚠️ Duplicate transcript blocked: {transcript_data['text'][:30]}...")
#                     return
                
#                 # Check against recent transcripts in storage
#                 recent_transcripts = self.consultation_transcripts.get(consultation_id, [])[-5:]
#                 for recent in recent_transcripts:
#                     if (recent['text'].strip() == transcript_data['text'].strip() and 
#                         recent['speaker'] == transcript_data['speaker']):
#                         logger.warning(f"⚠️ Duplicate in history blocked: {transcript_data['text'][:30]}...")
#                         return
                
#                 # Update last hash
#                 self.last_transcript_hash[consultation_id] = transcript_hash
                
#                 # Send to client
#                 await websocket.send_json(transcript_data)
#                 logger.info(f"✅ Sent to client: [{transcript_data['speaker']}] {transcript_data['text'][:40]}...")
                
#                 # Store only if final
#                 if transcript_data.get("is_final", False):
#                     self.consultation_transcripts[consultation_id].append({
#                         "text": transcript_data["text"],
#                         "original_text": transcript_data.get("original_text", transcript_data["text"]),
#                         "speaker": transcript_data["speaker"],
#                         "timestamp": transcript_data.get("timestamp", datetime.utcnow().isoformat()),
#                         "confidence": transcript_data.get("confidence", 0.95),
#                         "language": transcript_data.get("language", "en")
#                     })
#                     logger.info(f"💾 Stored final transcript: [{transcript_data['speaker']}] {transcript_data['text'][:40]}...")
                    
#             except Exception as e:
#                 logger.error(f"Error sending transcript: {e}")
#                 self.disconnect(consultation_id)
                
#     async def broadcast_status(self, consultation_id: str, status: str, message: str = ""):
#         """Send status update to the connected client"""
#         if consultation_id in self.active_connections:
#             websocket = self.active_connections[consultation_id]
#             try:
#                 await websocket.send_json({
#                     "type": "status",
#                     "status": status,
#                     "message": message,
#                     "timestamp": datetime.utcnow().isoformat()
#                 })
#             except Exception as e:
#                 logger.error(f"Error sending status: {e}")
                
#     def get_transcript(self, consultation_id: str) -> List[dict]:
#         """Retrieve stored transcript for a consultation"""
#         return self.consultation_transcripts.get(consultation_id, [])
    
#     def clear_transcript(self, consultation_id: str):
#         """Clear transcript data for a consultation"""
#         if consultation_id in self.consultation_transcripts:
#             del self.consultation_transcripts[consultation_id]
#         if consultation_id in self.consultation_languages:
#             del self.consultation_languages[consultation_id]
#         if consultation_id in self.last_transcript_hash:
#             del self.last_transcript_hash[consultation_id]

# manager = ConnectionManager()


# from fastapi import WebSocket
# from typing import Dict, List, Optional
# import asyncio
# import json
# from datetime import datetime
# import logging
# import time
# import hashlib

# logger = logging.getLogger(__name__)

# class ConnectionManager:
#     """Manages WebSocket connections with aggressive duplicate prevention"""
    
#     def __init__(self):
#         self.active_connections: Dict[str, WebSocket] = {}
#         self.consultation_transcripts: Dict[str, List[dict]] = {}
#         self.consultation_languages: Dict[str, str] = {}
#         self.sent_hashes: Dict[str, set] = {}  # Track sent transcript hashes per consultation
#         self.last_send_time: Dict[str, float] = {}  # Track last send time per consultation
        
#     async def connect(self, consultation_id: str, websocket: WebSocket, language: str = "en"):
#         """Accept and store a new WebSocket connection"""
#         await websocket.accept()
#         self.active_connections[consultation_id] = websocket
#         self.consultation_transcripts[consultation_id] = []
#         self.consultation_languages[consultation_id] = language
#         self.sent_hashes[consultation_id] = set()
#         self.last_send_time[consultation_id] = 0
#         logger.info(f"✅ WebSocket connected: {consultation_id} (lang: {language})")
        
#     def disconnect(self, consultation_id: str):
#         """Remove a WebSocket connection"""
#         if consultation_id in self.active_connections:
#             del self.active_connections[consultation_id]
#         if consultation_id in self.consultation_languages:
#             del self.consultation_languages[consultation_id]
#         if consultation_id in self.sent_hashes:
#             del self.sent_hashes[consultation_id]
#         if consultation_id in self.last_send_time:
#             del self.last_send_time[consultation_id]
#         logger.info(f"🔌 WebSocket disconnected: {consultation_id}")
        
#     def set_language(self, consultation_id: str, language: str):
#         """Update the language for a consultation"""
#         self.consultation_languages[consultation_id] = language
        
#     def get_language(self, consultation_id: str) -> str:
#         """Get the language for a consultation"""
#         return self.consultation_languages.get(consultation_id, "en")
    
#     def _create_hash(self, text: str, speaker: str) -> str:
#         """Create hash for duplicate detection"""
#         normalized = text.strip().lower()
#         return hashlib.md5(f"{normalized}:{speaker}".encode()).hexdigest()
    
#     def _is_similar_text(self, text1: str, text2: str) -> bool:
#         """Check if two texts are too similar (fuzzy duplicate detection)"""
#         t1 = text1.strip().lower()
#         t2 = text2.strip().lower()
        
#         # Exact match
#         if t1 == t2:
#             return True
        
#         # One contains the other (substring)
#         if t1 in t2 or t2 in t1:
#             return True
        
#         # Very similar length and content
#         if abs(len(t1) - len(t2)) < 5:
#             words1 = set(t1.split())
#             words2 = set(t2.split())
#             overlap = len(words1 & words2) / max(len(words1), len(words2), 1)
#             if overlap > 0.8:  # 80% word overlap
#                 return True
        
#         return False
        
#     async def send_transcript(self, consultation_id: str, transcript_data: dict):
#         """Send transcript with AGGRESSIVE duplicate prevention"""
#         if consultation_id not in self.active_connections:
#             logger.warning(f"⚠️ No active connection for {consultation_id}")
#             return
        
#         websocket = self.active_connections[consultation_id]
        
#         try:
#             text = transcript_data['text'].strip()
#             speaker = transcript_data['speaker']
            
#             # CHECK 1: Rate limiting (min 0.3s between sends)
#             current_time = time.time()
#             last_time = self.last_send_time.get(consultation_id, 0)
#             if current_time - last_time < 0.3:
#                 logger.warning(f"🚫 Rate limit: {text[:30]}")
#                 return
            
#             # CHECK 2: Hash-based duplicate
#             transcript_hash = self._create_hash(text, speaker)
#             if transcript_hash in self.sent_hashes.get(consultation_id, set()):
#                 logger.warning(f"🚫 Hash duplicate: {text[:30]}")
#                 return
            
#             # CHECK 3: Check against recent transcripts (fuzzy match)
#             recent_transcripts = self.consultation_transcripts.get(consultation_id, [])[-10:]
#             for recent in recent_transcripts:
#                 if recent['speaker'] == speaker and self._is_similar_text(recent['text'], text):
#                     logger.warning(f"🚫 Fuzzy duplicate: {text[:30]}")
#                     return
            
#             # CHECK 4: Empty or very short text
#             if len(text) < 2:
#                 logger.warning(f"🚫 Text too short: '{text}'")
#                 return
            
#             # All checks passed - send it!
#             await websocket.send_json(transcript_data)
#             logger.info(f"✅ WebSocket SENT: [{speaker}] {text[:50]}")
            
#             # Update tracking
#             self.sent_hashes[consultation_id].add(transcript_hash)
#             self.last_send_time[consultation_id] = current_time
            
#             # Clear old hashes if too many
#             if len(self.sent_hashes[consultation_id]) > 200:
#                 old_hashes = list(self.sent_hashes[consultation_id])[:100]
#                 self.sent_hashes[consultation_id] = set(old_hashes)
            
#             # Store final transcripts
#             if transcript_data.get("is_final", False):
#                 self.consultation_transcripts[consultation_id].append({
#                     "text": text,
#                     "original_text": transcript_data.get("original_text", text),
#                     "speaker": speaker,
#                     "timestamp": transcript_data.get("timestamp", datetime.utcnow().isoformat()),
#                     "confidence": transcript_data.get("confidence", 0.95),
#                     "language": transcript_data.get("language", "en")
#                 })
#                 logger.info(f"💾 Stored: [{speaker}] {text[:40]}")
                    
#         except Exception as e:
#             logger.error(f"❌ Error sending transcript: {e}")
#             self.disconnect(consultation_id)
                
#     async def broadcast_status(self, consultation_id: str, status: str, message: str = ""):
#         """Send status update"""
#         if consultation_id in self.active_connections:
#             websocket = self.active_connections[consultation_id]
#             try:
#                 await websocket.send_json({
#                     "type": "status",
#                     "status": status,
#                     "message": message,
#                     "timestamp": datetime.utcnow().isoformat()
#                 })
#             except Exception as e:
#                 logger.error(f"Error sending status: {e}")
                
#     def get_transcript(self, consultation_id: str) -> List[dict]:
#         """Retrieve stored transcript"""
#         return self.consultation_transcripts.get(consultation_id, [])
    
#     def clear_transcript(self, consultation_id: str):
#         """Clear transcript data"""
#         if consultation_id in self.consultation_transcripts:
#             del self.consultation_transcripts[consultation_id]
#         if consultation_id in self.consultation_languages:
#             del self.consultation_languages[consultation_id]
#         if consultation_id in self.sent_hashes:
#             del self.sent_hashes[consultation_id]
#         if consultation_id in self.last_send_time:
#             del self.last_send_time[consultation_id]

# manager = ConnectionManager()

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