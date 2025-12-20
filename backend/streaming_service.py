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
from typing import Callable, Iterator
import logging
from datetime import datetime
import threading
import queue
from dotenv import load_dotenv
from os import getenv
import time
import hashlib
import numpy as np
import io
import wave

# Fix for torchaudio backend deprecation - import before pyannote
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

# Conditionally import pyannote
try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except Exception as e:
    PYANNOTE_AVAILABLE = False
    Pipeline = None
    print(f"Warning: pyannote.audio not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
ASSEMBLYAI_API_KEY = getenv("ASSEMBLYAI_API_KEY")
HUGGINGFACE_TOKEN = getenv("HUGGINGFACE_TOKEN")


class TranscriptionService:
    """
    Hybrid transcription service using:
    - AssemblyAI for speech-to-text
    - Pyannote.audio for real-time speaker diarization
    """
    
    def __init__(self, api_key: str, consultation_id: str):
        self.api_key = ASSEMBLYAI_API_KEY
        
        if not self.api_key or self.api_key.strip() == "":
            raise ValueError("AssemblyAI API key is empty!")
        
        # Initialize diarization pipeline only if pyannote is available
        if not PYANNOTE_AVAILABLE:
            logger.warning("⚠️ Pyannote.audio not available - speaker diarization will use fallback")
            self.diarization_pipeline = None
        elif not HUGGINGFACE_TOKEN or HUGGINGFACE_TOKEN.strip() == "":
            logger.warning("⚠️ HuggingFace token missing - speaker diarization will use fallback")
            self.diarization_pipeline = None
        else:
            try:
                logger.info("🔄 Loading pyannote speaker diarization model...")
                # Try new 'token' parameter first (for newer huggingface_hub versions)
                try:
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        token=HUGGINGFACE_TOKEN  # New parameter name
                    )
                except TypeError:
                    # Fallback to old 'use_auth_token' parameter
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=HUGGINGFACE_TOKEN
                    )
                logger.info("✅ Pyannote diarization model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load pyannote model: {e}")
                logger.warning("⚠️ Falling back to simple speaker alternation")
                self.diarization_pipeline = None
        
        self.consultation_id = consultation_id
        self.client = None
        self.audio_source = None
        self.streaming_thread = None
        self.is_running = False
        self.on_transcript = None
        self.session_id = None
        
        # Speaker tracking
        self.speaker_mapping = {}  # Maps pyannote speaker (SPEAKER_00, SPEAKER_01) to role
        self.first_speaker_assigned = False
        self.audio_buffer = []  # Buffer for diarization
        self.sample_rate = 16000
        
        # Turn continuation tracking
        self.last_turn_time = 0
        self.last_turn_speaker = None
        self.last_turn_length = 0
        self.turn_continuation_window = 2.0  # If next turn comes within 2s, might be same speaker
        
        # Turn buffering (to prevent excessive fragmentation)
        self.turn_buffer = []  # Buffer incomplete turns
        self.turn_buffer_speaker = None
        self.turn_buffer_start_time = 0
        self.buffer_timeout = 1.5  # Send buffered content after 1.5s of silence
        self.min_turn_length = 15  # Minimum characters before sending (increased from 3)
        
        # Duplicate prevention - increased capacity
        self.sent_hashes = set()
        self.recent_transcripts = []
        self.max_recent = 50  # Increased buffer
        self.last_processed_time = 0  # Cooldown tracking
        self.min_time_between_transcripts = 0.3  # Reduced to 0.3s (was too aggressive at 0.5s)
        
        logger.info(f"🔑 TranscriptionService initialized with hybrid diarization")
        
    def start(self, on_transcript_callback: Callable):
        """Start streaming transcription with speaker diarization"""
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
    
    def _is_likely_continuation(self, text: str, current_time: float) -> bool:
        """
        Detect if this turn is likely a CONTINUATION of the previous speaker
        Now works with buffering system
        """
        if not self.last_turn_speaker:
            return False
        
        time_since_last = current_time - self.last_turn_time
        
        # If it's been more than 4 seconds, probably a new speaker
        if time_since_last > 4.0:
            return False
        
        # If within continuation window (3 seconds) - extended for buffering
        if time_since_last <= 3.0:
            # Check if last turn was suspiciously short (likely cut off)
            if self.last_turn_length < 40:  # Less than 40 characters
                logger.info(f"🔄 Likely continuation: last turn was short ({self.last_turn_length} chars) and only {time_since_last:.1f}s ago")
                return True
            
            # Check if current turn is also short (fragment pattern)
            if len(text) < 50:
                logger.info(f"🔄 Likely continuation: current turn is short ({len(text)} chars) within {time_since_last:.1f}s")
                return True
        
        return False
    
    def _detect_speaker_from_audio(self, audio_chunk: bytes) -> str:
        """
        Use pyannote.audio to detect speaker from audio chunk
        Returns 'Doctor' or 'Patient'
        """
        if not self.diarization_pipeline or not PYANNOTE_AVAILABLE:
            # Fallback: alternate between speakers
            return self._fallback_speaker_detection()
        
        try:
            # Convert audio bytes to numpy array
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Pyannote expects mono audio
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Create a temporary audio structure
            import torch
            
            # Convert to torch tensor
            waveform = torch.from_numpy(audio_array).unsqueeze(0)
            
            # Run diarization on this chunk
            diarization = self.diarization_pipeline({
                "waveform": waveform, 
                "sample_rate": self.sample_rate
            })
            
            # Get the most active speaker in this chunk
            speaker_times = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                duration = turn.end - turn.start
                speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
            
            if speaker_times:
                # Get speaker with most talk time in this chunk
                active_speaker = max(speaker_times, key=speaker_times.get)
                return self._assign_speaker_role(active_speaker)
            
        except Exception as e:
            logger.error(f"❌ Diarization error: {e}")
        
        return self._fallback_speaker_detection()
    
    def _smart_speaker_detection(self, text: str, current_time: float) -> str:
        """
        Intelligently detect speaker considering turn continuation patterns
        """
        # Check if this is likely a continuation of previous speaker
        if self._is_likely_continuation(text, current_time):
            logger.info(f"✅ CONTINUATION: Keeping speaker as {self.last_turn_speaker}")
            return self.last_turn_speaker
        
        # Otherwise, detect speaker normally
        if len(self.audio_buffer) > 0:
            # Concatenate recent audio for diarization
            audio_data = b''.join(self.audio_buffer[-10:])
            speaker = self._detect_speaker_from_audio(audio_data)
            logger.info(f"🎯 Detected speaker: {speaker}")
            return speaker
        else:
            speaker = self._fallback_speaker_detection()
            logger.info(f"🎯 Fallback speaker: {speaker}")
            return speaker
    
    def _fallback_speaker_detection(self) -> str:
        """
        Improved fallback: Use context and timing to determine speaker
        Works with buffering system
        """
        # If we're actively buffering, keep same speaker
        if self.turn_buffer_speaker:
            return self.turn_buffer_speaker
        
        # If we have recent transcripts, be smart about it
        if len(self.recent_transcripts) > 0:
            last_speaker = self.recent_transcripts[-1][1]
            last_time = self.recent_transcripts[-1][2]
            current_time = time.time()
            time_diff = current_time - last_time
            
            # If it's been less than 3 seconds and last turn was short, 
            # likely same speaker continuing
            if time_diff < 3.0 and len(self.recent_transcripts[-1][0]) < 50:
                logger.info(f"🔄 Fallback continuation: Keeping {last_speaker} (short turn + quick follow-up)")
                return last_speaker
            
            # Otherwise, alternate speakers (normal conversation pattern)
            return "Patient" if last_speaker == "Doctor" else "Doctor"
        
        # Very first transcript - assume Doctor speaks first
        return "Doctor"
    
    def _assign_speaker_role(self, speaker_label: str) -> str:
        """
        Map pyannote speaker labels (SPEAKER_00, SPEAKER_01) to roles (Doctor, Patient)
        Assumes first speaker is Doctor
        """
        if speaker_label not in self.speaker_mapping:
            if not self.first_speaker_assigned:
                # First speaker is Doctor
                self.speaker_mapping[speaker_label] = "Doctor"
                self.first_speaker_assigned = True
                logger.info(f"🏥 Speaker {speaker_label} assigned as Doctor")
            else:
                # Second speaker is Patient
                self.speaker_mapping[speaker_label] = "Patient"
                logger.info(f"🤒 Speaker {speaker_label} assigned as Patient")
        
        return self.speaker_mapping[speaker_label]
    
    def _on_turn(self, client: StreamingClient, event: TurnEvent):
        """Handle turns with pyannote speaker detection"""
        
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
        
        logger.info(f"📥 Complete turn: '{text[:70]}...'")
        
        # FIRST: Check for hash duplicates (fastest check)
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()
        if text_hash in self.sent_hashes:
            logger.warning(f"🚫 DUPLICATE (hash) - BLOCKING: '{text[:50]}'")
            return
        
        # SECOND: Check for recent duplicates (before speaker detection)
        if self._is_recent_duplicate(text, ""):  # Pass empty speaker to ignore speaker in check
            logger.warning(f"🚫 DUPLICATE (recent) - BLOCKING: '{text[:50]}'")
            return
        
        # NOW: Determine speaker from audio buffer
        if len(self.audio_buffer) > 0:
            # Concatenate recent audio for diarization
            audio_data = b''.join(self.audio_buffer[-10:])  # Last 10 chunks (~1 second)
            speaker = self._detect_speaker_from_audio(audio_data)
            logger.info(f"🎯 Detected speaker: {speaker}")
        else:
            speaker = self._fallback_speaker_detection()
            logger.info(f"🎯 Fallback speaker: {speaker}")
        
        # Process and send (duplicates already checked above)
        self._process_transcript(text, speaker, text_hash)
    
    def _normalize_text_aggressive(self, text: str) -> str:
        """AGGRESSIVELY normalize text for duplicate detection"""
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove ALL punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Strip
        text = text.strip()
        
        return text
    
    def _is_recent_duplicate(self, text: str, speaker: str) -> bool:
        """AGGRESSIVE duplicate detection - blocks ANY similar text regardless of speaker"""
        # Normalize aggressively
        normalized = self._normalize_text_aggressive(text)
        
        # Skip empty after normalization
        if not normalized or len(normalized) < 3:
            return False
        
        current_time = time.time()
        
        for recent_text, recent_speaker, recent_time, _ in self.recent_transcripts:
            time_diff = current_time - recent_time
            
            # Check duplicates within last 15 seconds (increased window)
            if time_diff < 15.0:
                recent_normalized = self._normalize_text_aggressive(recent_text)
                
                # AGGRESSIVE MATCH 1: Exact normalized match
                if normalized == recent_normalized:
                    logger.warning(f"🚫 EXACT DUPLICATE: '{text[:50]}'")
                    return True
                
                # AGGRESSIVE MATCH 2: Substring match (even for shorter texts)
                if len(normalized) > 10:
                    if normalized in recent_normalized:
                        logger.warning(f"🚫 SUBSTRING DUPLICATE (current in recent): '{text[:50]}'")
                        return True
                    if recent_normalized in normalized:
                        logger.warning(f"🚫 SUBSTRING DUPLICATE (recent in current): '{text[:50]}'")
                        return True
                
                # AGGRESSIVE MATCH 3: Word overlap similarity
                words_current = set(normalized.split())
                words_recent = set(recent_normalized.split())
                
                if len(words_current) > 3 and len(words_recent) > 3:
                    # Calculate word overlap
                    overlap = len(words_current.intersection(words_recent))
                    total_words = max(len(words_current), len(words_recent))
                    word_similarity = overlap / total_words if total_words > 0 else 0
                    
                    if word_similarity > 0.75:  # 75% word overlap
                        logger.warning(f"🚫 WORD OVERLAP DUPLICATE: {word_similarity:.1%} similarity")
                        return True
                
                # AGGRESSIVE MATCH 4: Levenshtein-like similarity
                if len(normalized) > 5 and len(recent_normalized) > 5:
                    similarity = self._calculate_similarity_ratio(normalized, recent_normalized)
                    if similarity > 0.80:  # 80% character similarity
                        logger.warning(f"🚫 CHARACTER SIMILARITY DUPLICATE: {similarity:.1%}")
                        return True
        
        return False
    
    def _calculate_similarity_ratio(self, str1: str, str2: str) -> float:
        """Calculate similarity ratio between two strings"""
        if str1 == str2:
            return 1.0
        
        if not str1 or not str2:
            return 0.0
        
        # Use set of character bigrams for better similarity detection
        def get_bigrams(string):
            return set(string[i:i+2] for i in range(len(string) - 1))
        
        bigrams1 = get_bigrams(str1)
        bigrams2 = get_bigrams(str2)
        
        if not bigrams1 or not bigrams2:
            # Fallback to simple character overlap
            longer = max(len(str1), len(str2))
            matches = sum(1 for a, b in zip(str1, str2) if a == b)
            return matches / longer if longer > 0 else 0.0
        
        # Jaccard similarity
        intersection = len(bigrams1.intersection(bigrams2))
        union = len(bigrams1.union(bigrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def _process_transcript(self, text: str, speaker: str, text_hash_value: str):
        """Process and send transcript with aggressive duplicate tracking"""
        
        current_time = time.time()
        
        # Store hash and transcript
        self.sent_hashes.add(text_hash_value)
        self.recent_transcripts.append((text, speaker, current_time, text_hash_value))
        
        # Cleanup old data (keep more history for better duplicate detection)
        if len(self.recent_transcripts) > self.max_recent:
            removed = self.recent_transcripts.pop(0)
            logger.debug(f"🗑️ Removed old: '{removed[0][:30]}'")
        
        if len(self.sent_hashes) > 300:
            # Keep more hashes in memory
            recent_hashes = {h for _, _, _, h in self.recent_transcripts[-150:]}
            self.sent_hashes = recent_hashes
            logger.debug(f"🧹 Hash cleanup: kept {len(self.sent_hashes)}")
        
        # Clear old transcripts beyond time window
        cutoff_time = current_time - 20.0  # Keep 20 seconds of history
        self.recent_transcripts = [
            t for t in self.recent_transcripts 
            if t[2] > cutoff_time
        ]
        
        logger.info(f"✅ APPROVED: [{speaker}] '{text[:60]}' | History: {len(self.recent_transcripts)} | Duration: {len(text)} chars")
        
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
        """Send audio to stream and buffer for diarization"""
        if self.audio_source and self.is_running:
            try:
                # Send to AssemblyAI
                self.audio_source.add_audio(audio_data)
                
                # Buffer for speaker detection
                self.audio_buffer.append(audio_data)
                if len(self.audio_buffer) > 50:  # Keep last ~5 seconds
                    self.audio_buffer.pop(0)
                    
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


class SummaryService:
    """Generate clinical summaries"""
    
    def __init__(self, api_key: str):
        self.api_key = ASSEMBLYAI_API_KEY
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
