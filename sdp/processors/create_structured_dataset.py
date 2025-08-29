"""
Create structured dataset with WAV/JSON pairs for HuggingFace upload.
Based on scripts/create_dataset.py but integrated into the SDP pipeline.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import soundfile as sf
import numpy as np

from sdp.logging import logger
from sdp.processors.base_processor import BaseProcessor


class CreateStructuredDataset(BaseProcessor):
    """
    Create a structured dataset from manifest with WAV/JSON pairs.
    Speech segments only - no dialogue entries.
    
    Output structure:
    - file/speech/{device_id}/{session_id}/segment_XXXX.wav
    - file/speech/{device_id}/{session_id}/segment_XXXX.json
    """
    
    def __init__(
        self,
        output_dir: str,
        input_manifest_file: Optional[str] = None,
        output_manifest_file: Optional[str] = None,
        audio_filepath_key: str = "audio_filepath",
        offset_key: str = "offset",
        duration_key: str = "duration",
        text_key: str = "text",
        should_run: bool = True,
        **kwargs,
    ):
        super().__init__(output_manifest_file=output_manifest_file, **kwargs)
        
        self.output_dir = Path(output_dir)
        self.input_manifest_file = input_manifest_file
        self.audio_filepath_key = audio_filepath_key
        self.offset_key = offset_key
        self.duration_key = duration_key
        self.text_key = text_key
        self.should_run = should_run
        
        # Create output directory for speech segments
        if self.should_run:
            (self.output_dir / "file" / "speech").mkdir(parents=True, exist_ok=True)
    
    def process(self):
        """Process manifest and create structured dataset."""
        
        if not self.should_run:
            logger.info("Skipping CreateStructuredDataset processor (should_run=False)")
            return
        
        # Load input manifest
        if self.input_manifest_file:
            manifest_data = self._load_manifest(self.input_manifest_file)
        else:
            manifest_data = self.data
        
        # Group entries by session
        sessions = self._group_by_session(manifest_data)
        
        # Process each session
        processed_entries = []
        for session_id, session_data in sessions.items():
            device_id = session_data["device_id"]
            segments = session_data["segments"]
            
            logger.info(f"Processing session {session_id} with {len(segments)} segments")
            
            # Create speech segment files
            for idx, segment in enumerate(segments):
                entry = self._create_speech_segment(
                    segment, idx, device_id, session_id
                )
                if entry:
                    processed_entries.append(entry)
        
        # Create dataset metadata
        self._create_dataset_metadata(len(sessions), len(processed_entries))
        
        # Save output manifest
        if self.output_manifest_file:
            self._save_manifest(processed_entries, self.output_manifest_file)
        
        logger.info(
            f"Created structured dataset with {len(processed_entries)} segments "
            f"from {len(sessions)} sessions"
        )
        
        self.data = processed_entries
    
    def _load_manifest(self, manifest_file: str) -> List[Dict]:
        """Load manifest from JSON lines file."""
        entries = []
        with open(manifest_file, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries
    
    def _save_manifest(self, entries: List[Dict], manifest_file: str) -> None:
        """Save manifest to JSON lines file."""
        Path(manifest_file).parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
    
    def _parse_audio_path(self, audio_path: str) -> tuple:
        """Extract device_id and session_id from audio filepath."""
        # Example: ej_au1200UD262024_11_5f288992-a3fe-4f67-8089-130e62cb7592.wav
        filename = Path(audio_path).stem
        parts = filename.split('_')
        
        if len(parts) >= 4:
            # Device ID is first part with year
            device_id = f"{parts[0]}_{parts[1]}"  # ej_au1200UD262024_11
            # Session ID is the UUID at the end
            session_id = parts[-1]  # 5f288992-a3fe-4f67-8089-130e62cb7592
            return device_id, session_id
        
        # Fallback
        return "unknown_device", filename
    
    def _group_by_session(self, manifest_data: List[Dict]) -> Dict:
        """Group manifest entries by session."""
        sessions = {}
        
        for entry in manifest_data:
            audio_path = entry.get(self.audio_filepath_key, "")
            device_id, session_id = self._parse_audio_path(audio_path)
            
            if session_id not in sessions:
                sessions[session_id] = {
                    "device_id": device_id,
                    "segments": []
                }
            
            sessions[session_id]["segments"].append(entry)
        
        return sessions
    
    def _create_speech_segment(
        self, 
        segment: Dict, 
        idx: int, 
        device_id: str, 
        session_id: str
    ) -> Optional[Dict]:
        """Create a speech segment with WAV and JSON files."""
        
        # Create output directory
        segment_dir = self.output_dir / "file" / "speech" / device_id / session_id
        segment_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        segment_filename = f"{device_id}_{session_id}_segment_{idx:04d}"
        
        # Extract audio segment
        audio_path = segment.get(self.audio_filepath_key)
        offset = segment.get(self.offset_key, 0.0)
        duration = segment.get(self.duration_key, 0.0)
        
        if audio_path and Path(audio_path).exists():
            try:
                # Load and slice audio
                audio_data, sample_rate = sf.read(audio_path)
                
                # Calculate frame boundaries
                start_frame = int(offset * sample_rate)
                end_frame = int((offset + duration) * sample_rate)
                
                # Handle both mono and stereo
                if len(audio_data.shape) == 1:
                    segment_audio = audio_data[start_frame:end_frame]
                else:
                    segment_audio = audio_data[start_frame:end_frame, :]
                
                # Save WAV file
                wav_path = segment_dir / f"{segment_filename}.wav"
                sf.write(wav_path, segment_audio, sample_rate)
                
                # Prepare metadata
                segment_metadata = {
                    "text": segment.get(self.text_key, ""),
                    "duration": duration,
                    "offset": offset,
                    "segment_id": segment.get("segment_id", idx),
                    "source_lang": segment.get("source_lang", "en"),
                    "speaker": segment.get("speaker", "unknown"),
                    "sid": session_id,
                    "device_id": device_id,
                    "is_speech_segment": segment.get("is_speech_segment", True),
                    "emotion": segment.get("emotion", "<|emo:undefined|>"),
                    "pnc": segment.get("pnc", "pnc"),
                    "itn": segment.get("itn", "itn"),
                    "timestamp": segment.get("timestamp", "notimestamp"),
                    "diarize": segment.get("diarize", "nodiarize"),
                    "audio_filepath": str(wav_path),
                    "original_audio": audio_path,
                }
                
                # Save JSON metadata
                json_path = segment_dir / f"{segment_filename}.json"
                with open(json_path, 'w') as f:
                    json.dump(segment_metadata, f, indent=2)
                
                # Return updated entry for manifest
                updated_entry = segment.copy()
                updated_entry["structured_audio_filepath"] = str(wav_path)
                updated_entry["structured_json_filepath"] = str(json_path)
                updated_entry["session_id"] = session_id
                updated_entry["device_id"] = device_id
                
                return updated_entry
                
            except Exception as e:
                logger.warning(
                    f"Failed to create segment {segment_filename}: {str(e)}"
                )
                return None
        else:
            logger.warning(f"Audio file not found: {audio_path}")
            return None
    def _create_dataset_metadata(self, num_sessions: int, num_segments: int) -> None:
        """Create dataset metadata file."""
        
        metadata = {
            "dataset_name": "EJ_AU_DriveThru_Dataset",
            "description": "Drive-thru audio conversations with transcriptions - speech segments only",
            "language": "en",
            "task": "speech_recognition",
            "statistics": {
                "total_sessions": num_sessions,
                "total_segments": num_segments,
            },
            "audio_format": {
                "sample_rate": 16000,
                "channels": 1,
                "format": "wav",
            },
            "processing": {
                "transcription_model": "whisper-large-v3",
                "punctuation_model": "Qwen3-8B",
                "language_detection_threshold": 0.7,
                "min_duration": 0.5,
                "max_duration": 90.0,
            },
            "structure": {
                "speech_segments": "file/speech/{device_id}/{session_id}/",
            }
        }
        
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create README
        readme_content = f"""# EJ AU Drive-Thru Dataset (Speech Segments)

## Dataset Statistics
- Total Sessions: {num_sessions}
- Total Segments: {num_segments}

## Structure
- `file/speech/`: Individual speech segments with WAV audio and JSON metadata
  - Organization: `speech/device_id/session_id/`
  - Files: `segment_XXXX.wav` and `segment_XXXX.json`

## Audio Format
- Sample Rate: 16 kHz
- Channels: 1 (mono)
- Format: WAV

## Usage
Each speech segment includes:
- `.wav`: Audio file (16kHz mono, properly sliced)
- `.json`: Metadata with transcription and timing information

## Metadata Fields
- `text`: Transcription with punctuation
- `duration`: Segment duration in seconds
- `offset`: Start time in original audio
- `segment_id`: Unique identifier
- `session_id`: Session UUID
- `device_id`: Recording device
- Additional NLP tags (emotion, pnc, itn, etc.)

## Processing Pipeline
1. Audio conversion to 16kHz mono WAV
2. Language detection (English filter) 
3. Transcription with Whisper Large-v3
4. Punctuation restoration with Qwen3-8B
5. Quality filtering and validation
6. Speech segment extraction
"""
        
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.info(f"Created dataset metadata: {metadata_path}")