"""Enhanced cross-channel validation with audio bleeding correction."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import logging

from rapidfuzz import fuzz
from tqdm import tqdm
import soundfile as sf

from sdp.processors.base_processor import BaseProcessor

logger = logging.getLogger(__name__)


class EnhancedCrossChannelValidation(BaseProcessor):
    """
    Enhanced validation and correction using synchronized dual-channel audio.
    
    Context: Drive-thru audio systems record two synchronized channels:
    - mic.wav: Customer microphone (often noisy, may have employee bleeding)
    - spk.wav: Employee speaker (cleaner, direct capture)
    Both files have identical duration and share the same timeline.
    
    This processor:
    1. Matches ASR segments from synchronized recordings (same timeline)
    2. Detects audio bleeding (employee voice in customer mic)
    3. Uses cleaner spk channel for correction when appropriate
    4. Handles overlapping speech (both talking simultaneously)
    """
    
    def __init__(
        self,
        output_manifest_file: str,
        input_manifest_file: str,
        speaker_manifest_file: Optional[str] = None,
        text_field: str = "text",
        validated_field: str = "text_validated",
        confidence_field: str = "cross_channel_confidence",
        bleeding_field: str = "audio_bleeding_detected",
        correction_source_field: str = "correction_source",
        similarity_threshold: float = 0.7,
        bleeding_detection_threshold: float = 0.8,
        time_tolerance: float = 0.5,  # seconds tolerance for timestamp matching
        prefer_speaker_channel: bool = True,
        min_speaker_duration: float = 1.0,  # minimum duration to consider speaker channel
        **kwargs,
    ):
        """
        Initialize EnhancedCrossChannelValidation processor.
        
        Args:
            input_manifest_file: Input manifest path (mic channel)
            output_manifest_file: Output manifest path
            speaker_manifest_file: Path to speaker channel manifest
            text_field: Field containing transcription text
            validated_field: Field to store validated text
            confidence_field: Field to store cross-channel confidence
            bleeding_field: Field to indicate audio bleeding detection
            correction_source_field: Field to indicate correction source (mic/spk/merged)
            similarity_threshold: Minimum similarity for validation (0-1)
            bleeding_detection_threshold: Threshold for detecting audio bleeding
            time_tolerance: Tolerance in seconds for timestamp matching
            prefer_speaker_channel: Always prefer speaker channel when available
            min_speaker_duration: Minimum duration to use speaker channel
        """
        super().__init__(output_manifest_file, input_manifest_file, **kwargs)
        
        self.speaker_manifest_file = speaker_manifest_file
        self.text_field = text_field
        self.validated_field = validated_field
        self.confidence_field = confidence_field
        self.bleeding_field = bleeding_field
        self.correction_source_field = correction_source_field
        self.similarity_threshold = similarity_threshold
        self.bleeding_detection_threshold = bleeding_detection_threshold
        self.time_tolerance = time_tolerance
        self.prefer_speaker_channel = prefer_speaker_channel
        self.min_speaker_duration = min_speaker_duration
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'bleeding_detected': 0,
            'speaker_corrections': 0,
            'mic_only': 0,
            'speaker_only': 0,
            'merged': 0,
            'high_confidence': 0,
            'low_confidence': 0
        }
        
        # Load speaker manifest if provided
        self.speaker_segments = {}
        if speaker_manifest_file and os.path.exists(speaker_manifest_file):
            self._load_speaker_manifest()
    
    def _load_speaker_manifest(self):
        """Load and index speaker channel manifest by session and timestamp."""
        logger.info(f"Loading speaker manifest from {self.speaker_manifest_file}")
        
        with open(self.speaker_manifest_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                session_id = entry.get('session_id', '')
                
                if session_id not in self.speaker_segments:
                    self.speaker_segments[session_id] = []
                
                self.speaker_segments[session_id].append({
                    'offset': entry.get('offset', 0),
                    'duration': entry.get('duration', 0),
                    'text': entry.get(self.text_field, ''),
                    'audio_filepath': entry.get('audio_filepath', ''),
                    'segment_id': entry.get('segment_id', 0)
                })
        
        # Sort segments by offset for efficient searching
        for session_id in self.speaker_segments:
            self.speaker_segments[session_id].sort(key=lambda x: x['offset'])
        
        logger.info(f"Loaded {sum(len(segs) for segs in self.speaker_segments.values())} speaker segments")
    
    def _find_matching_speaker_segment(self, session_id: str, offset: float, duration: float) -> Optional[Dict]:
        """
        Find speaker segment that matches the given timestamp.
        
        IMPORTANT: Both mic.wav and spk.wav are from the same recording session with:
        - Identical total duration (perfectly synchronized)
        - Same timeline (offset=10s means 10s into both files)
        - Simultaneous recording (no time drift between channels)
        
        The challenge is matching ASR segments that may have different boundaries
        due to overlapping speech or different voice activity detection.
        
        Args:
            session_id: Session identifier (both channels share same session)
            offset: Start time in the synchronized recording
            duration: Duration of mic segment
            
        Returns:
            Matching speaker segment or None
        """
        if session_id not in self.speaker_segments:
            return None
        
        mic_end = offset + duration
        
        # Look for speaker segments at the same time period
        # Since recordings are synchronized, we're looking for segments
        # that occur at the same moment in the timeline
        for spk_segment in self.speaker_segments[session_id]:
            spk_offset = spk_segment['offset']
            spk_duration = spk_segment['duration']
            spk_end = spk_offset + spk_duration
            
            # Check for temporal overlap
            # Tolerance accounts for ASR segmentation differences, not time sync issues
            overlap_start = max(offset - self.time_tolerance, spk_offset)
            overlap_end = min(mic_end + self.time_tolerance, spk_end)
            
            if overlap_end > overlap_start:
                overlap_ratio = (overlap_end - overlap_start) / duration
                
                # Require significant overlap (>50%)
                # This handles cases where speakers talk over each other
                if overlap_ratio > 0.5:
                    return spk_segment
        
        return None
    
    def _detect_audio_bleeding(self, mic_text: str, spk_text: str) -> Tuple[bool, float]:
        """
        Detect if mic channel contains audio bleeding from speaker.
        
        Args:
            mic_text: Transcription from mic channel
            spk_text: Transcription from speaker channel
            
        Returns:
            Tuple of (bleeding_detected, confidence_score)
        """
        if not mic_text or not spk_text:
            return False, 0.0
        
        # Calculate similarity
        similarity = fuzz.ratio(mic_text.lower(), spk_text.lower()) / 100.0
        
        # Check for substring matches (common in bleeding)
        mic_words = set(mic_text.lower().split())
        spk_words = set(spk_text.lower().split())
        
        if len(spk_words) > 0:
            word_overlap = len(mic_words & spk_words) / len(spk_words)
        else:
            word_overlap = 0
        
        # Bleeding likely if high similarity or significant word overlap
        bleeding_score = max(similarity, word_overlap)
        bleeding_detected = bleeding_score >= self.bleeding_detection_threshold
        
        return bleeding_detected, bleeding_score
    
    def _correct_with_speaker_channel(self, mic_text: str, spk_text: str, bleeding_score: float) -> Tuple[str, str, float]:
        """
        Correct mic transcription using speaker channel.
        
        Args:
            mic_text: Original mic transcription
            spk_text: Speaker channel transcription
            bleeding_score: Audio bleeding confidence score
            
        Returns:
            Tuple of (corrected_text, source, confidence)
        """
        if not spk_text:
            return mic_text, 'mic', 0.5
        
        # If high bleeding detected, use speaker channel
        if bleeding_score >= self.bleeding_detection_threshold:
            return spk_text, 'spk_bleeding', bleeding_score
        
        # If speaker channel is significantly longer/cleaner, prefer it
        if self.prefer_speaker_channel and len(spk_text.split()) > len(mic_text.split()) * 1.5:
            return spk_text, 'spk_preferred', 0.8
        
        # Calculate similarity for validation
        similarity = fuzz.ratio(mic_text.lower(), spk_text.lower()) / 100.0
        
        # If very similar, use speaker (typically cleaner)
        if similarity >= 0.9:
            return spk_text, 'spk_validated', similarity
        
        # If moderately similar, merge intelligently
        if similarity >= self.similarity_threshold:
            merged_text = self._merge_transcriptions(mic_text, spk_text)
            return merged_text, 'merged', similarity
        
        # Otherwise keep mic transcription
        return mic_text, 'mic', similarity
    
    def _merge_transcriptions(self, mic_text: str, spk_text: str) -> str:
        """
        Intelligently merge two transcriptions.
        
        Args:
            mic_text: Mic channel transcription
            spk_text: Speaker channel transcription
            
        Returns:
            Merged transcription
        """
        # Simple strategy: use longer transcription as base
        if len(spk_text) > len(mic_text):
            return spk_text
        return mic_text
    
    def process_entry(self, entry: Dict) -> Optional[Dict]:
        """Process a single manifest entry with cross-channel validation."""
        
        self.stats['total_processed'] += 1
        
        # Get mic channel info
        mic_text = entry.get(self.text_field, '')
        session_id = entry.get('session_id', '')
        offset = entry.get('offset', 0)
        duration = entry.get('duration', 0)
        
        # Find matching speaker segment
        spk_segment = self._find_matching_speaker_segment(session_id, offset, duration)
        
        if spk_segment and spk_segment['text']:
            spk_text = spk_segment['text']
            
            # Detect audio bleeding
            bleeding_detected, bleeding_score = self._detect_audio_bleeding(mic_text, spk_text)
            
            if bleeding_detected:
                self.stats['bleeding_detected'] += 1
            
            # Correct using speaker channel
            corrected_text, source, confidence = self._correct_with_speaker_channel(
                mic_text, spk_text, bleeding_score
            )
            
            # Update statistics
            if source.startswith('spk'):
                self.stats['speaker_corrections'] += 1
            elif source == 'merged':
                self.stats['merged'] += 1
            
            if confidence >= 0.8:
                self.stats['high_confidence'] += 1
            else:
                self.stats['low_confidence'] += 1
            
            # Update entry
            entry[self.validated_field] = corrected_text
            entry[self.confidence_field] = confidence
            entry[self.bleeding_field] = bleeding_detected
            entry[self.correction_source_field] = source
            
            # Add debug info
            if bleeding_detected or source != 'mic':
                entry['validation_metadata'] = {
                    'original_mic': mic_text,
                    'speaker_text': spk_text,
                    'bleeding_score': bleeding_score,
                    'time_alignment': {
                        'mic_offset': offset,
                        'mic_duration': duration,
                        'spk_offset': spk_segment['offset'],
                        'spk_duration': spk_segment['duration']
                    }
                }
        else:
            # No speaker segment found
            self.stats['mic_only'] += 1
            entry[self.validated_field] = mic_text
            entry[self.confidence_field] = 0.5
            entry[self.bleeding_field] = False
            entry[self.correction_source_field] = 'mic_only'
        
        return entry
    
    def process(self):
        """Main processing method."""
        
        with open(self.input_manifest_file, 'r') as f_in, \
             open(self.output_manifest_file, 'w') as f_out:
            
            lines = f_in.readlines()
            for line in tqdm(lines, desc="Enhanced cross-channel validation"):
                entry = json.loads(line)
                processed_entry = self.process_entry(entry)
                if processed_entry:
                    f_out.write(json.dumps(processed_entry) + '\n')
        
        # Report statistics
        self.finalize({})
    
    def finalize(self, metrics: Dict) -> None:
        """Finalize processing and report statistics."""
        
        print("\n=== Enhanced Cross-Channel Validation Statistics ===")
        print(f"Total entries processed: {self.stats['total_processed']}")
        print(f"Audio bleeding detected: {self.stats['bleeding_detected']}")
        print(f"Speaker channel corrections: {self.stats['speaker_corrections']}")
        print(f"Mic-only segments: {self.stats['mic_only']}")
        print(f"Merged transcriptions: {self.stats['merged']}")
        
        if self.stats['total_processed'] > 0:
            bleeding_rate = (self.stats['bleeding_detected'] / self.stats['total_processed']) * 100
            correction_rate = (self.stats['speaker_corrections'] / self.stats['total_processed']) * 100
            print(f"\nBleeding detection rate: {bleeding_rate:.2f}%")
            print(f"Speaker correction rate: {correction_rate:.2f}%")
            
        print(f"\nConfidence distribution:")
        print(f"  High confidence (>= 0.8): {self.stats['high_confidence']}")
        print(f"  Low confidence (< 0.8): {self.stats['low_confidence']}")