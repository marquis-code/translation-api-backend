
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
from typing import Callable, Optional, Iterator
import logging
from datetime import datetime
import threading
import queue
from dotenv import load_dotenv
from os import getenv
import time
import hashlib
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
HARDCODED_API_KEY = getenv("ASSEMBLYAI_API_KEY")


def normalize_text(text: str) -> str:
    """
    Aggressive text normalization for duplicate detection
    """
    # Convert to lowercase
    text = text.lower()
    # Remove all punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()


def text_hash(text: str) -> str:
    """
    Create hash of normalized text for fast duplicate detection
    """
    normalized = normalize_text(text)
    return hashlib.md5(normalized.encode()).hexdigest()


class ImprovedTurnBasedDiarizer:
    """
    IMPROVED: Turn-based speaker detection with smart switching logic
    """
    
    def __init__(self):
        self.current_speaker = "Doctor"  # Doctor speaks first
        self.last_switch_time = time.time()
        self.min_turn_duration = 2.5  # Minimum 2.5s before allowing switch
        self.last_text_hash = ""
        self.consecutive_same_speaker = 0
        
        logger.info("✅ Improved turn-based diarizer initialized")
    
    def should_switch_speaker(self, text: str) -> bool:
        """
        Determine if we should switch to the other speaker
        """
        current_time = time.time()
        time_since_last_switch = current_time - self.last_switch_time
        
        # Get normalized hash
        current_hash = text_hash(text)
        
        # NEVER switch if text hash is same as last (it's a duplicate)
        if current_hash == self.last_text_hash:
            logger.info(f"🔒 Keeping same speaker - duplicate text detected")
            return False
        
        # Must wait minimum duration before switching
        if time_since_last_switch < self.min_turn_duration:
            logger.info(f"🔒 Keeping same speaker - too soon to switch ({time_since_last_switch:.1f}s)")
            self.consecutive_same_speaker += 1
            return False
        
        # After 2 consecutive turns with sufficient time, switch
        if self.consecutive_same_speaker >= 2:
            logger.info(f"🔄 Forcing speaker switch - {self.consecutive_same_speaker} consecutive turns")
            return True
        
        # Default: switch after sufficient time
        logger.info(f"🔄 Switching speaker - sufficient time passed ({time_since_last_switch:.1f}s)")
        return True
    
    def get_speaker_for_turn(self, text: str) -> str:
        """
        Get speaker for current turn
        """
        current_time = time.time()
        current_hash = text_hash(text)
        
        # Check if we should switch
        if self.should_switch_speaker(text):
            # Switch speaker
            self.current_speaker = "Patient" if self.current_speaker == "Doctor" else "Doctor"
            self.last_switch_time = current_time
            self.consecutive_same_speaker = 0
        else:
            # Keep same speaker
            self.consecutive_same_speaker += 1
        
        # Update last text hash
        self.last_text_hash = current_hash
        
        return self.current_speaker


class AudioStreamSource:
    """Audio source for streaming"""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_active = True
        
    def __iter__(self) -> Iterator[bytes]:
        """Yield audio chunks"""
        while self.is_active:
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                if audio_chunk is not None:
                    yield audio_chunk
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Audio stream error: {e}")
                break
    
    def add_audio(self, audio_data: bytes):
        """Add audio to queue"""
        if self.is_active:
            self.audio_queue.put(audio_data)
    
    def stop(self):
        """Stop streaming"""
        self.is_active = False
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break


class TranscriptionService:
    """Real-time transcription with aggressive duplicate prevention"""
    
    def __init__(self, api_key: str, consultation_id: str):
        self.api_key = HARDCODED_API_KEY
        
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("API key is empty!")
        
        self.consultation_id = consultation_id
        self.client = None
        self.audio_source = None
        self.streaming_thread = None
        self.is_running = False
        self.on_transcript = None
        self.session_id = None
        
        # Improved turn-based diarizer
        self.diarizer = ImprovedTurnBasedDiarizer()
        
        # ULTRA AGGRESSIVE duplicate prevention using hashes
        self.sent_hashes = set()  # Track hashes of ALL sent transcripts
        self.recent_transcripts = []  # Keep recent for detailed logging
        self.max_recent = 20
        
        logger.info(f"🔑 TranscriptionService initialized with ULTRA-AGGRESSIVE duplicate prevention")
        
    def start(self, on_transcript_callback: Callable):
        """Start streaming transcription"""
        self.is_running = True
        self.on_transcript = on_transcript_callback
        
        try:
            logger.info(f"🚀 Creating StreamingClient...")
            
            self.client = StreamingClient(
                StreamingClientOptions(
                    api_key=self.api_key,
                    api_host="streaming.assemblyai.com",
                )
            )
            
            self.client.on(StreamingEvents.Begin, self._on_begin)
            self.client.on(StreamingEvents.Turn, self._on_turn)
            self.client.on(StreamingEvents.Termination, self._on_terminated)
            self.client.on(StreamingEvents.Error, self._on_error)
            
            logger.info("🔌 Connecting to AssemblyAI...")
            self.client.connect(
                StreamingParameters(
                    sample_rate=16000,
                    format_turns=True,
                )
            )
            
            self.audio_source = AudioStreamSource()
            
            self.streaming_thread = threading.Thread(
                target=self._stream_audio,
                daemon=True
            )
            self.streaming_thread.start()
            
            logger.info(f"✅ Transcription service started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start: {e}", exc_info=True)
            self.is_running = False
            raise
    
    def _stream_audio(self):
        """Stream audio thread"""
        try:
            logger.info("🎤 Audio streaming started...")
            self.client.stream(self.audio_source)
            logger.info("🎤 Audio streaming ended")
        except Exception as e:
            logger.error(f"❌ Audio streaming error: {e}", exc_info=True)
            self.is_running = False
    
    def _on_begin(self, client: StreamingClient, event: BeginEvent):
        """Session started"""
        self.session_id = event.id
        logger.info(f"✅ Session started: {event.id}")
    
    def _on_turn(self, client: StreamingClient, event: TurnEvent):
        """Handle turns with ULTRA-AGGRESSIVE duplicate filtering"""
        
        if not event.transcript or not event.transcript.strip():
            return
        
        text = event.transcript.strip()
        end_of_turn = event.end_of_turn
        
        # Only process complete turns
        if not end_of_turn:
            return
        
        # Skip very short transcripts
        if len(text) < 3:
            logger.info(f"⏩ Skipping very short text: '{text}'")
            return
        
        logger.info(f"📥 Complete turn received: '{text[:60]}...'")
        
        # STEP 1: Check hash-based duplicate (FASTEST, MOST RELIABLE)
        current_hash = text_hash(text)
        if current_hash in self.sent_hashes:
            logger.warning(f"🚫 DUPLICATE DETECTED (hash match) - BLOCKING: '{text[:50]}'")
            return
        
        # STEP 2: Get speaker for THIS turn
        speaker = self.diarizer.get_speaker_for_turn(text)
        
        # STEP 3: Additional safety check - look for similar recent transcripts
        if self._is_recent_duplicate(text, speaker):
            logger.warning(f"🚫 DUPLICATE DETECTED (recent match) - BLOCKING: '{text[:50]}'")
            return
        
        # STEP 4: Process transcript (it's unique!)
        self._process_transcript(text, speaker, current_hash)
    
    def _is_recent_duplicate(self, text: str, speaker: str) -> bool:
        """
        Check if text is similar to recent transcripts
        This is a backup check in case hash matching fails
        """
        normalized = normalize_text(text)
        current_time = time.time()
        
        for recent_text, recent_speaker, recent_time, recent_hash in self.recent_transcripts:
            time_diff = current_time - recent_time
            
            # Check within last 5 seconds
            if time_diff < 5.0:
                recent_normalized = normalize_text(recent_text)
                
                # Exact normalized match
                if normalized == recent_normalized:
                    logger.warning(f"  → Found exact normalized match from {time_diff:.1f}s ago")
                    return True
                
                # Substring match (one contains the other)
                if len(normalized) > 20:  # Only for longer texts
                    if normalized in recent_normalized or recent_normalized in normalized:
                        logger.warning(f"  → Found substring match from {time_diff:.1f}s ago")
                        return True
        
        return False
    
    def _process_transcript(self, text: str, speaker: str, text_hash_value: str):
        """Process and send transcript"""
        
        current_time = time.time()
        
        # Add hash to sent set
        self.sent_hashes.add(text_hash_value)
        
        # Add to recent transcripts
        self.recent_transcripts.append((text, speaker, current_time, text_hash_value))
        
        # Cleanup old transcripts (keep last 20)
        if len(self.recent_transcripts) > self.max_recent:
            self.recent_transcripts.pop(0)
        
        # Cleanup sent_hashes if it gets too large (keep last 200)
        if len(self.sent_hashes) > 200:
            # Keep only hashes from recent transcripts
            recent_hashes = {h for _, _, _, h in self.recent_transcripts[-100:]}
            self.sent_hashes = recent_hashes
        
        logger.info(f"✅ APPROVED for sending: [{speaker}] '{text[:50]}'")
        
        transcript_data = {
            "type": "transcript",
            "text": text,
            "speaker": speaker,
            "timestamp": datetime.utcnow().isoformat(),
            "confidence": 0.95,
            "is_final": True,
        }
        
        if self.on_transcript:
            try:
                self.on_transcript(transcript_data)
                logger.info(f"📤 SENT: [{speaker}] {text[:60]}")
            except Exception as e:
                logger.error(f"❌ Callback error: {e}")
    
    def _on_terminated(self, client: StreamingClient, event: TerminationEvent):
        """Session terminated"""
        logger.info(f"🔌 Session ended: {event.audio_duration_seconds}s")
        self.is_running = False
    
    def _on_error(self, client: StreamingClient, error: StreamingError):
        """Streaming error"""
        logger.error(f"❌ STREAMING ERROR: {error}")
        self.is_running = False
    
    def send_audio(self, audio_data: bytes):
        """Send audio to stream"""
        if self.audio_source and self.is_running:
            try:
                self.audio_source.add_audio(audio_data)
            except Exception as e:
                logger.error(f"❌ Error sending audio: {e}")
                self.is_running = False
    
    def stop(self):
        """Stop service"""
        logger.info("🛑 Stopping...")
        self.is_running = False
        
        if self.audio_source:
            self.audio_source.stop()
        
        if self.client:
            try:
                self.client.disconnect(terminate=True)
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
        
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=2.0)
        
        logger.info(f"🛑 Stopped")


class SummaryService:
    """Generate clinical summaries"""
    
    def __init__(self, api_key: str):
        self.api_key = HARDCODED_API_KEY
        aai.settings.api_key = self.api_key
    
    async def generate_summary(self, transcript: str, segments: list, target_language: str = "en") -> dict:
        """Generate summary"""
        formatted_transcript = "\n".join([
            f"{seg['speaker']}: {seg.get('original_text', seg['text'])}" for seg in segments
        ])
        
        try:
            summary = self._extract_structured_summary(formatted_transcript, segments, target_language)
            return summary
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise
    
    def _extract_structured_summary(self, transcript: str, segments: list, target_language: str) -> dict:
        """Extract structured info"""
        
        identifiers = self._extract_identifiers(segments)
        history = self._extract_history(segments)
        examination = self._extract_examination(segments)
        diagnoses = self._extract_diagnoses(segments)
        treatment = self._extract_treatment(segments)
        advice = self._extract_advice(segments)
        next_steps = self._extract_next_steps(segments)
        
        if target_language != "en":
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='en', target=target_language)
            
            try:
                identifiers = translator.translate(identifiers)
                history = translator.translate(history)
                examination = translator.translate(examination)
                diagnoses = translator.translate(diagnoses)
                treatment = translator.translate(treatment)
                advice = translator.translate(advice)
                next_steps = translator.translate(next_steps)
            except Exception as e:
                logger.error(f"Translation error: {e}")
        
        return {
            "identifiers": identifiers,
            "history": history,
            "examination": examination,
            "diagnoses": diagnoses,
            "treatment": treatment,
            "advice": advice,
            "next_steps": next_steps,
            "language": target_language
        }
    
    def _extract_identifiers(self, segments: list) -> str:
        return "Name: [To be filled]\nAge: [To be filled]\nSex: [To be filled]\nLocation: [To be filled]"
    
    def _extract_history(self, segments: list) -> str:
        keywords = ["complain", "pain", "symptom", "problem", "feel", "since"]
        relevant = [f"- {s.get('original_text', s['text'])}" for s in segments 
                   if any(k in s.get('original_text', s['text']).lower() for k in keywords)][:8]
        return f"Chief Complaint:\n" + "\n".join(relevant) if relevant else "Chief Complaint: [To be documented]"
    
    def _extract_examination(self, segments: list) -> str:
        keywords = ["blood pressure", "temperature", "pulse", "examine"]
        relevant = [f"- {s.get('original_text', s['text'])}" for s in segments 
                   if any(k in s.get('original_text', s['text']).lower() for k in keywords) 
                   and s['speaker'] == 'Doctor'][:8]
        return f"Examination:\n" + "\n".join(relevant) if relevant else "Examination: [To be documented]"
    
    def _extract_diagnoses(self, segments: list) -> str:
        keywords = ["diagnos", "appears", "likely", "condition"]
        relevant = [f"- {s.get('original_text', s['text'])}" for s in segments 
                   if any(k in s.get('original_text', s['text']).lower() for k in keywords) 
                   and s['speaker'] == 'Doctor'][:6]
        return f"Assessment:\n" + "\n".join(relevant) if relevant else "Assessment: [To be determined]"
    
    def _extract_treatment(self, segments: list) -> str:
        keywords = ["prescribe", "medication", "tablet", "treatment"]
        relevant = [f"- {s.get('original_text', s['text'])}" for s in segments 
                   if any(k in s.get('original_text', s['text']).lower() for k in keywords) 
                   and s['speaker'] == 'Doctor'][:8]
        return f"Treatment:\n" + "\n".join(relevant) if relevant else "Treatment: [To be prescribed]"
    
    def _extract_advice(self, segments: list) -> str:
        keywords = ["advise", "recommend", "should", "avoid", "diet"]
        relevant = [f"- {s.get('original_text', s['text'])}" for s in segments 
                   if any(k in s.get('original_text', s['text']).lower() for k in keywords) 
                   and s['speaker'] == 'Doctor'][:8]
        return f"Advice:\n" + "\n".join(relevant) if relevant else "Advice: [To be discussed]"
    
    def _extract_next_steps(self, segments: list) -> str:
        keywords = ["follow", "appointment", "return", "test"]
        relevant = [f"- {s.get('original_text', s['text'])}" for s in segments 
                   if any(k in s.get('original_text', s['text']).lower() for k in keywords) 
                   and s['speaker'] == 'Doctor'][:6]
        return f"Follow-up:\n" + "\n".join(relevant) if relevant else "Follow-up: Schedule in 1-2 weeks"


# import assemblyai as aai
# from assemblyai.streaming.v3 import (
#     BeginEvent,
#     StreamingClient,
#     StreamingClientOptions,
#     StreamingError,
#     StreamingEvents,
#     StreamingParameters,
#     TerminationEvent,
#     TurnEvent,
# )
# from typing import Callable, Optional, Iterator
# import logging
# from datetime import datetime
# import threading
# import queue
# from dotenv import load_dotenv
# from os import getenv
# import time
# import hashlib
# import re

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# load_dotenv()
# HARDCODED_API_KEY = getenv("ASSEMBLYAI_API_KEY")


# def normalize_text(text: str) -> str:
#     """Aggressive text normalization for duplicate detection"""
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     text = ' '.join(text.split())
#     return text.strip()


# def text_hash(text: str) -> str:
#     """Create hash of normalized text for fast duplicate detection"""
#     normalized = normalize_text(text)
#     return hashlib.md5(normalized.encode()).hexdigest()


# def is_question(text: str) -> bool:
#     """Detect if text is a question"""
#     text_lower = text.lower().strip()
    
#     # Check for question marks
#     if '?' in text:
#         return True
    
#     # Check for question words at start
#     question_words = ['what', 'when', 'where', 'who', 'why', 'how', 'can', 'could', 
#                       'would', 'should', 'do', 'does', 'did', 'is', 'are', 'was', 'were']
#     first_word = text_lower.split()[0] if text_lower.split() else ""
    
#     return first_word in question_words


# def is_short_response(text: str) -> bool:
#     """Detect if text is a short response (likely from patient)"""
#     word_count = len(text.split())
    
#     # Very short responses (1-5 words)
#     if word_count <= 5:
#         return True
    
#     # Common short responses
#     short_responses = ['yes', 'no', 'okay', 'ok', 'sure', 'yeah', 'right', 
#                        'got it', 'i understand', 'makes sense']
#     text_lower = text.lower().strip()
    
#     return any(resp in text_lower for resp in short_responses) and word_count < 15


# class SmartDiarizer:
#     """
#     SMART APPROACH: Detect speaker changes based on:
#     1. Silence/pause duration between utterances
#     2. Speech patterns (questions vs answers)
#     3. Speech length (short vs long)
#     4. Context awareness
#     """
    
#     def __init__(self):
#         self.current_speaker = "Doctor"  # Doctor typically starts
#         self.last_utterance_time = time.time()
#         self.last_text = ""
#         self.turn_history = []
        
#         # Thresholds
#         self.min_pause_for_switch = 1.5  # 1.5 second pause suggests speaker change
#         self.long_pause_threshold = 3.0  # 3+ seconds definitely means speaker change
        
#         logger.info("✅ Smart context-aware diarizer initialized")
    
#     def get_speaker_for_turn(self, text: str) -> str:
#         """
#         Intelligently determine speaker based on context
#         """
#         current_time = time.time()
#         pause_duration = current_time - self.last_utterance_time
        
#         # Analyze the text
#         is_quest = is_question(text)
#         is_short = is_short_response(text)
        
#         logger.info(f"📊 Analysis: pause={pause_duration:.1f}s, question={is_quest}, short={is_short}")
        
#         # RULE 1: Very long pause (3+ seconds) = speaker definitely changed
#         if pause_duration >= self.long_pause_threshold:
#             self._switch_speaker()
#             logger.info(f"🔄 SWITCH (long pause: {pause_duration:.1f}s)")
        
#         # RULE 2: Short response after question = patient responding
#         elif is_short and self._last_was_question():
#             if self.current_speaker == "Doctor":
#                 self._switch_speaker()
#                 logger.info(f"🔄 SWITCH (short response to question)")
        
#         # RULE 3: Question after patient response = doctor asking
#         elif is_quest and self.current_speaker == "Patient":
#             if pause_duration >= self.min_pause_for_switch:
#                 self._switch_speaker()
#                 logger.info(f"🔄 SWITCH (question after patient)")
        
#         # RULE 4: Medium pause (1.5-3s) + different speech pattern = likely switch
#         elif pause_duration >= self.min_pause_for_switch:
#             if self._should_switch_based_on_pattern(text, is_quest, is_short):
#                 self._switch_speaker()
#                 logger.info(f"🔄 SWITCH (pattern change with pause)")
        
#         # RULE 5: Same speaker continuing (no switch)
#         else:
#             logger.info(f"➡️ CONTINUE (same speaker)")
        
#         # Update state
#         self.last_utterance_time = current_time
#         self.last_text = text
#         self.turn_history.append({
#             "speaker": self.current_speaker,
#             "text": text[:50],
#             "is_question": is_quest,
#             "is_short": is_short,
#             "timestamp": current_time
#         })
        
#         # Keep last 10 turns
#         if len(self.turn_history) > 10:
#             self.turn_history.pop(0)
        
#         return self.current_speaker
    
#     def _switch_speaker(self):
#         """Switch to the other speaker"""
#         self.current_speaker = "Patient" if self.current_speaker == "Doctor" else "Doctor"
    
#     def _last_was_question(self) -> bool:
#         """Check if last utterance was a question"""
#         if len(self.turn_history) == 0:
#             return False
#         return self.turn_history[-1].get("is_question", False)
    
#     def _should_switch_based_on_pattern(self, text: str, is_quest: bool, is_short: bool) -> bool:
#         """
#         Determine if pattern suggests speaker change
#         """
#         if len(self.turn_history) == 0:
#             return False
        
#         last_turn = self.turn_history[-1]
        
#         # Doctor was asking questions, now getting longer response = patient
#         if last_turn.get("is_question") and not is_quest and not is_short:
#             return True
        
#         # Patient gave short response, now getting question = doctor
#         if last_turn.get("is_short") and is_quest:
#             return True
        
#         # Pattern mismatch suggests change
#         return False


# class AudioStreamSource:
#     """Audio source for streaming"""
    
#     def __init__(self):
#         self.audio_queue = queue.Queue()
#         self.is_active = True
        
#     def __iter__(self) -> Iterator[bytes]:
#         """Yield audio chunks"""
#         while self.is_active:
#             try:
#                 audio_chunk = self.audio_queue.get(timeout=0.1)
#                 if audio_chunk is not None:
#                     yield audio_chunk
#             except queue.Empty:
#                 continue
#             except Exception as e:
#                 logger.error(f"Audio stream error: {e}")
#                 break
    
#     def add_audio(self, audio_data: bytes):
#         """Add audio to queue"""
#         if self.is_active:
#             self.audio_queue.put(audio_data)
    
#     def stop(self):
#         """Stop streaming"""
#         self.is_active = False
#         while not self.audio_queue.empty():
#             try:
#                 self.audio_queue.get_nowait()
#             except queue.Empty:
#                 break


# class TranscriptionService:
#     """Real-time transcription with smart context-aware diarization"""
    
#     def __init__(self, api_key: str, consultation_id: str):
#         self.api_key = HARDCODED_API_KEY
        
#         if not self.api_key or self.api_key.strip() == "":
#             raise ValueError("API key is empty!")
        
#         self.consultation_id = consultation_id
#         self.client = None
#         self.audio_source = None
#         self.streaming_thread = None
#         self.is_running = False
#         self.on_transcript = None
#         self.session_id = None
        
#         # Smart context-aware diarizer
#         self.diarizer = SmartDiarizer()
        
#         # Ultra aggressive duplicate prevention
#         self.sent_hashes = set()
#         self.recent_transcripts = []
#         self.max_recent = 20
        
#         logger.info(f"🔑 TranscriptionService initialized with smart diarization")
        
#     def start(self, on_transcript_callback: Callable):
#         """Start streaming transcription"""
#         self.is_running = True
#         self.on_transcript = on_transcript_callback
        
#         try:
#             logger.info(f"🚀 Creating StreamingClient...")
            
#             self.client = StreamingClient(
#                 StreamingClientOptions(
#                     api_key=self.api_key,
#                     api_host="streaming.assemblyai.com",
#                 )
#             )
            
#             self.client.on(StreamingEvents.Begin, self._on_begin)
#             self.client.on(StreamingEvents.Turn, self._on_turn)
#             self.client.on(StreamingEvents.Termination, self._on_terminated)
#             self.client.on(StreamingEvents.Error, self._on_error)
            
#             logger.info("🔌 Connecting to AssemblyAI...")
#             self.client.connect(
#                 StreamingParameters(
#                     sample_rate=16000,
#                     format_turns=True,
#                 )
#             )
            
#             self.audio_source = AudioStreamSource()
            
#             self.streaming_thread = threading.Thread(
#                 target=self._stream_audio,
#                 daemon=True
#             )
#             self.streaming_thread.start()
            
#             logger.info(f"✅ Transcription service started")
            
#         except Exception as e:
#             logger.error(f"❌ Failed to start: {e}", exc_info=True)
#             self.is_running = False
#             raise
    
#     def _stream_audio(self):
#         """Stream audio thread"""
#         try:
#             logger.info("🎤 Audio streaming started...")
#             self.client.stream(self.audio_source)
#             logger.info("🎤 Audio streaming ended")
#         except Exception as e:
#             logger.error(f"❌ Audio streaming error: {e}", exc_info=True)
#             self.is_running = False
    
#     def _on_begin(self, client: StreamingClient, event: BeginEvent):
#         """Session started"""
#         self.session_id = event.id
#         logger.info(f"✅ Session started: {event.id}")
    
#     def _on_turn(self, client: StreamingClient, event: TurnEvent):
#         """Handle turns with smart context-aware diarization"""
        
#         if not event.transcript or not event.transcript.strip():
#             return
        
#         text = event.transcript.strip()
#         end_of_turn = event.end_of_turn
        
#         # Only process complete turns
#         if not end_of_turn:
#             return
        
#         # Skip very short transcripts
#         if len(text) < 3:
#             logger.info(f"⏩ Skipping very short text: '{text}'")
#             return
        
#         logger.info(f"📥 Complete turn: '{text[:70]}...'")
        
#         # STEP 1: Hash-based duplicate check
#         current_hash = text_hash(text)
#         if current_hash in self.sent_hashes:
#             logger.warning(f"🚫 DUPLICATE (hash) - BLOCKING: '{text[:50]}'")
#             return
        
#         # STEP 2: Smart speaker detection based on context
#         speaker = self.diarizer.get_speaker_for_turn(text)
        
#         # STEP 3: Additional duplicate check
#         if self._is_recent_duplicate(text, speaker):
#             logger.warning(f"🚫 DUPLICATE (recent) - BLOCKING: '{text[:50]}'")
#             return
        
#         # STEP 4: Process and send
#         self._process_transcript(text, speaker, current_hash)
    
#     def _is_recent_duplicate(self, text: str, speaker: str) -> bool:
#         """Check if text is similar to recent transcripts"""
#         normalized = normalize_text(text)
#         current_time = time.time()
        
#         for recent_text, recent_speaker, recent_time, recent_hash in self.recent_transcripts:
#             time_diff = current_time - recent_time
            
#             if time_diff < 5.0:
#                 recent_normalized = normalize_text(recent_text)
                
#                 # Exact normalized match
#                 if normalized == recent_normalized:
#                     return True
                
#                 # Substring match for longer texts
#                 if len(normalized) > 20:
#                     if normalized in recent_normalized or recent_normalized in normalized:
#                         return True
        
#         return False
    
#     def _process_transcript(self, text: str, speaker: str, text_hash_value: str):
#         """Process and send transcript"""
        
#         current_time = time.time()
        
#         # Add hash to sent set
#         self.sent_hashes.add(text_hash_value)
        
#         # Add to recent transcripts
#         self.recent_transcripts.append((text, speaker, current_time, text_hash_value))
        
#         # Cleanup
#         if len(self.recent_transcripts) > self.max_recent:
#             self.recent_transcripts.pop(0)
        
#         if len(self.sent_hashes) > 200:
#             recent_hashes = {h for _, _, _, h in self.recent_transcripts[-100:]}
#             self.sent_hashes = recent_hashes
        
#         logger.info(f"✅ APPROVED: [{speaker}] '{text[:50]}'")
        
#         transcript_data = {
#             "type": "transcript",
#             "text": text,
#             "speaker": speaker,
#             "timestamp": datetime.utcnow().isoformat(),
#             "confidence": 0.95,
#             "is_final": True,
#         }
        
#         if self.on_transcript:
#             try:
#                 self.on_transcript(transcript_data)
#                 logger.info(f"📤 SENT: [{speaker}] {text[:60]}")
#             except Exception as e:
#                 logger.error(f"❌ Callback error: {e}")
    
#     def _on_terminated(self, client: StreamingClient, event: TerminationEvent):
#         """Session terminated"""
#         logger.info(f"🔌 Session ended: {event.audio_duration_seconds}s")
#         self.is_running = False
    
#     def _on_error(self, client: StreamingClient, error: StreamingError):
#         """Streaming error"""
#         logger.error(f"❌ STREAMING ERROR: {error}")
#         self.is_running = False
    
#     def send_audio(self, audio_data: bytes):
#         """Send audio to stream"""
#         if self.audio_source and self.is_running:
#             try:
#                 self.audio_source.add_audio(audio_data)
#             except Exception as e:
#                 logger.error(f"❌ Error sending audio: {e}")
#                 self.is_running = False
    
#     def stop(self):
#         """Stop service"""
#         logger.info("🛑 Stopping...")
#         self.is_running = False
        
#         if self.audio_source:
#             self.audio_source.stop()
        
#         if self.client:
#             try:
#                 self.client.disconnect(terminate=True)
#             except Exception as e:
#                 logger.error(f"Error disconnecting: {e}")
        
#         if self.streaming_thread and self.streaming_thread.is_alive():
#             self.streaming_thread.join(timeout=2.0)
        
#         logger.info(f"🛑 Stopped")


# class SummaryService:
#     """Generate clinical summaries"""
    
#     def __init__(self, api_key: str):
#         self.api_key = HARDCODED_API_KEY
#         aai.settings.api_key = self.api_key
    
#     async def generate_summary(self, transcript: str, segments: list, target_language: str = "en") -> dict:
#         """Generate summary"""
#         formatted_transcript = "\n".join([
#             f"{seg['speaker']}: {seg.get('original_text', seg['text'])}" for seg in segments
#         ])
        
#         try:
#             summary = self._extract_structured_summary(formatted_transcript, segments, target_language)
#             return summary
#         except Exception as e:
#             logger.error(f"Error generating summary: {e}")
#             raise
    
#     def _extract_structured_summary(self, transcript: str, segments: list, target_language: str) -> dict:
#         """Extract structured info"""
        
#         identifiers = self._extract_identifiers(segments)
#         history = self._extract_history(segments)
#         examination = self._extract_examination(segments)
#         diagnoses = self._extract_diagnoses(segments)
#         treatment = self._extract_treatment(segments)
#         advice = self._extract_advice(segments)
#         next_steps = self._extract_next_steps(segments)
        
#         if target_language != "en":
#             from deep_translator import GoogleTranslator
#             translator = GoogleTranslator(source='en', target=target_language)
            
#             try:
#                 identifiers = translator.translate(identifiers)
#                 history = translator.translate(history)
#                 examination = translator.translate(examination)
#                 diagnoses = translator.translate(diagnoses)
#                 treatment = translator.translate(treatment)
#                 advice = translator.translate(advice)
#                 next_steps = translator.translate(next_steps)
#             except Exception as e:
#                 logger.error(f"Translation error: {e}")
        
#         return {
#             "identifiers": identifiers,
#             "history": history,
#             "examination": examination,
#             "diagnoses": diagnoses,
#             "treatment": treatment,
#             "advice": advice,
#             "next_steps": next_steps,
#             "language": target_language
#         }
    
#     def _extract_identifiers(self, segments: list) -> str:
#         return "Name: [To be filled]\nAge: [To be filled]\nSex: [To be filled]\nLocation: [To be filled]"
    
#     def _extract_history(self, segments: list) -> str:
#         keywords = ["complain", "pain", "symptom", "problem", "feel", "since"]
#         relevant = [f"- {s.get('original_text', s['text'])}" for s in segments 
#                    if any(k in s.get('original_text', s['text']).lower() for k in keywords)][:8]
#         return f"Chief Complaint:\n" + "\n".join(relevant) if relevant else "Chief Complaint: [To be documented]"
    
#     def _extract_examination(self, segments: list) -> str:
#         keywords = ["blood pressure", "temperature", "pulse", "examine"]
#         relevant = [f"- {s.get('original_text', s['text'])}" for s in segments 
#                    if any(k in s.get('original_text', s['text']).lower() for k in keywords) 
#                    and s['speaker'] == 'Doctor'][:8]
#         return f"Examination:\n" + "\n".join(relevant) if relevant else "Examination: [To be documented]"
    
#     def _extract_diagnoses(self, segments: list) -> str:
#         keywords = ["diagnos", "appears", "likely", "condition"]
#         relevant = [f"- {s.get('original_text', s['text'])}" for s in segments 
#                    if any(k in s.get('original_text', s['text']).lower() for k in keywords) 
#                    and s['speaker'] == 'Doctor'][:6]
#         return f"Assessment:\n" + "\n".join(relevant) if relevant else "Assessment: [To be determined]"
    
#     def _extract_treatment(self, segments: list) -> str:
#         keywords = ["prescribe", "medication", "tablet", "treatment"]
#         relevant = [f"- {s.get('original_text', s['text'])}" for s in segments 
#                    if any(k in s.get('original_text', s['text']).lower() for k in keywords) 
#                    and s['speaker'] == 'Doctor'][:8]
#         return f"Treatment:\n" + "\n".join(relevant) if relevant else "Treatment: [To be prescribed]"
    
#     def _extract_advice(self, segments: list) -> str:
#         keywords = ["advise", "recommend", "should", "avoid", "diet"]
#         relevant = [f"- {s.get('original_text', s['text'])}" for s in segments 
#                    if any(k in s.get('original_text', s['text']).lower() for k in keywords) 
#                    and s['speaker'] == 'Doctor'][:8]
#         return f"Advice:\n" + "\n".join(relevant) if relevant else "Advice: [To be discussed]"
    
#     def _extract_next_steps(self, segments: list) -> str:
#         keywords = ["follow", "appointment", "return", "test"]
#         relevant = [f"- {s.get('original_text', s['text'])}" for s in segments 
#                    if any(k in s.get('original_text', s['text']).lower() for k in keywords) 
#                    and s['speaker'] == 'Doctor'][:6]
#         return f"Follow-up:\n" + "\n".join(relevant) if relevant else "Follow-up: Schedule in 1-2 weeks"