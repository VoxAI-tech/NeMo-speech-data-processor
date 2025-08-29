#!/usr/bin/env python3
"""
WebDataset Creator for Drive-thru Audio Pipeline
Creates WebDataset tar archives with session-aware sharding and proper structure
"""

import json
import os
import tarfile
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import logging

from sdp.processors.base_processor import BaseProcessor
from sdp.utils.common import write_manifest
from sdp.logging import logger

class CreateDriveThruWebDataset(BaseProcessor):
    """
    Create WebDataset tar archives from processed manifests with session integrity.
    
    Unlike the standard ConvertToTarredAudioDataset, this processor:
    1. Preserves session structure (keeps related segments together)
    2. Maintains device/location/timestamp metadata
    3. Creates proper WebDataset structure with .wav and .json pairs
    4. Supports channel separation (mic/spk)
    """
    
    def __init__(
        self,
        output_dir: str,
        num_shards: int = 16,
        max_duration_per_shard: float = 3600,  # 1 hour per shard
        maintain_session_integrity: bool = True,
        shuffle_sessions: bool = True,
        output_format: str = "wav",
        include_metadata: bool = True,
        min_duration: float = 0.5,
        max_duration: float = 20.0,
        **kwargs
    ):
        """
        Initialize WebDataset creator.
        
        Args:
            output_dir: Directory to save WebDataset tar files
            num_shards: Number of tar shards to create
            max_duration_per_shard: Maximum audio duration per shard in seconds
            maintain_session_integrity: Keep all segments from same session together
            shuffle_sessions: Shuffle sessions before sharding
            output_format: Audio format in tar files (wav/flac)
            include_metadata: Include JSON metadata with each audio file
            min_duration: Minimum segment duration to include
            max_duration: Maximum segment duration to include
        """
        super().__init__(**kwargs)
        
        self.output_dir = Path(output_dir)
        self.num_shards = num_shards
        self.max_duration_per_shard = max_duration_per_shard
        self.maintain_session_integrity = maintain_session_integrity
        self.shuffle_sessions = shuffle_sessions
        self.output_format = output_format
        self.include_metadata = include_metadata
        self.min_duration = min_duration
        self.max_duration = max_duration
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _parse_manifest_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse manifest entry to extract session information.
        
        Expected fields from pipeline:
        - audio_filepath: Path to audio file
        - offset: Start time in seconds
        - duration: Duration in seconds
        - text: Transcription
        - segment_id: Unique segment identifier
        - source_lang: Language code
        - Additional metadata fields
        """
        # Extract session info from audio path if available
        audio_path = Path(entry.get("audio_filepath", ""))
        
        # Try to parse hierarchical structure
        # Expected: .../country/location/device_id/year/month/day/hour/session_id/file.wav
        parts = audio_path.parts
        
        session_info = {
            "audio_filepath": str(audio_path),
            "offset": entry.get("offset", 0),
            "duration": entry.get("duration", 0),
            "text": entry.get("text", ""),
            "segment_id": entry.get("segment_id", ""),
            "source_lang": entry.get("source_lang", "en"),
        }
        
        # Try to extract hierarchical metadata
        if len(parts) >= 9:
            try:
                # Navigate backwards from filename
                idx = -1
                if parts[idx].endswith(".wav"):
                    idx -= 1  # Skip filename
                    
                session_info.update({
                    "session_id": parts[idx] if idx >= -len(parts) else "",
                    "hour": parts[idx-1] if idx-1 >= -len(parts) else "",
                    "day": parts[idx-2] if idx-2 >= -len(parts) else "",
                    "month": parts[idx-3] if idx-3 >= -len(parts) else "",
                    "year": parts[idx-4] if idx-4 >= -len(parts) else "",
                    "device_id": parts[idx-5] if idx-5 >= -len(parts) else "",
                    "location": parts[idx-6] if idx-6 >= -len(parts) else "",
                    "country": parts[idx-7] if idx-7 >= -len(parts) else "",
                })
            except (IndexError, ValueError):
                # Fallback: generate session ID from segment_id
                session_info["session_id"] = entry.get("segment_id", "").split("_")[0] if entry.get("segment_id") else ""
        
        # Add any additional metadata fields
        for key, value in entry.items():
            if key not in session_info:
                session_info[key] = value
                
        return session_info
    
    def _group_by_session(self, manifest_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Group manifest entries by session ID."""
        sessions = defaultdict(list)
        
        for entry in manifest_data:
            parsed = self._parse_manifest_entry(entry)
            session_id = parsed.get("session_id", "unknown")
            
            # Filter by duration
            duration = parsed.get("duration", 0)
            if self.min_duration <= duration <= self.max_duration:
                sessions[session_id].append(parsed)
        
        # Sort segments within each session by offset
        for session_id in sessions:
            sessions[session_id].sort(key=lambda x: (x.get("offset", 0), x.get("segment_id", "")))
        
        return dict(sessions)
    
    def _create_shards(self, sessions: Dict[str, List[Dict]]) -> List[List[Dict]]:
        """
        Distribute sessions across shards.
        
        If maintain_session_integrity is True, keeps entire sessions together.
        Otherwise, can split sessions across shards.
        """
        shards = [[] for _ in range(self.num_shards)]
        shard_durations = [0.0] * self.num_shards
        
        # Get session list
        session_ids = list(sessions.keys())
        
        if self.shuffle_sessions:
            random.shuffle(session_ids)
        
        if self.maintain_session_integrity:
            # Keep sessions together
            for session_id in session_ids:
                session_segments = sessions[session_id]
                session_duration = sum(seg.get("duration", 0) for seg in session_segments)
                
                # Find shard with least duration
                min_shard_idx = shard_durations.index(min(shard_durations))
                
                # Add entire session to that shard
                shards[min_shard_idx].extend(session_segments)
                shard_durations[min_shard_idx] += session_duration
        else:
            # Can split sessions across shards
            all_segments = []
            for session_id in session_ids:
                all_segments.extend(sessions[session_id])
            
            if self.shuffle_sessions:
                random.shuffle(all_segments)
            
            # Distribute segments round-robin
            for i, segment in enumerate(all_segments):
                shard_idx = i % self.num_shards
                shards[shard_idx].append(segment)
                shard_durations[shard_idx] += segment.get("duration", 0)
        
        # Remove empty shards
        shards = [s for s in shards if s]
        
        logger.info(f"Created {len(shards)} shards with durations: {[f'{d:.1f}s' for d in shard_durations[:len(shards)]]}")
        
        return shards
    
    def _load_audio_segment(self, entry: Dict) -> Optional[bytes]:
        """Load audio segment from file with offset/duration slicing."""
        import soundfile as sf
        import numpy as np
        import io
        
        try:
            audio_path = entry["audio_filepath"]
            offset = entry.get("offset", 0)
            duration = entry.get("duration", None)
            
            # Load audio segment
            with sf.SoundFile(audio_path) as f:
                sample_rate = f.samplerate
                start_frame = int(offset * sample_rate)
                
                if duration:
                    num_frames = int(duration * sample_rate)
                else:
                    num_frames = -1  # Read to end
                
                # Seek to start position
                f.seek(start_frame)
                
                # Read audio data
                audio_data = f.read(frames=num_frames, dtype='float32')
                
                # Ensure mono
                if audio_data.ndim > 1:
                    audio_data = audio_data.mean(axis=1)
                
                # Convert to WAV bytes
                buffer = io.BytesIO()
                sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
                return buffer.getvalue()
                
        except Exception as e:
            logger.warning(f"Failed to load audio segment: {e}")
            return None
    
    def _write_tar_shard(self, shard_data: List[Dict], shard_idx: int) -> str:
        """Write a single tar shard with WebDataset format."""
        shard_path = self.output_dir / f"shard_{shard_idx:05d}.tar"
        manifest_entries = []
        
        logger.info(f"Writing shard {shard_idx} with {len(shard_data)} segments to {shard_path}")
        
        with tarfile.open(shard_path, "w") as tar:
            for i, entry in enumerate(shard_data):
                # Generate unique key for this sample
                sample_key = f"{shard_idx:05d}_{i:08d}"
                
                # Load audio data
                audio_bytes = self._load_audio_segment(entry)
                if audio_bytes is None:
                    continue
                
                # Add audio file to tar
                audio_name = f"{sample_key}.{self.output_format}"
                audio_info = tarfile.TarInfo(name=audio_name)
                audio_info.size = len(audio_bytes)
                tar.addfile(audio_info, io.BytesIO(audio_bytes))
                
                # Create metadata
                metadata = {
                    "text": entry.get("text", ""),
                    "duration": entry.get("duration", 0),
                    "offset": entry.get("offset", 0),
                    "segment_id": entry.get("segment_id", ""),
                    "session_id": entry.get("session_id", ""),
                    "device_id": entry.get("device_id", ""),
                    "source_lang": entry.get("source_lang", "en"),
                }
                
                # Add optional metadata fields
                for key in ["country", "location", "year", "month", "day", "hour", 
                           "speaker", "emotion", "pnc", "itn", "timestamp", "diarize"]:
                    if key in entry:
                        metadata[key] = entry[key]
                
                if self.include_metadata:
                    # Add JSON metadata to tar
                    json_name = f"{sample_key}.json"
                    json_bytes = json.dumps(metadata, ensure_ascii=False).encode('utf-8')
                    json_info = tarfile.TarInfo(name=json_name)
                    json_info.size = len(json_bytes)
                    tar.addfile(json_info, io.BytesIO(json_bytes))
                
                # Add to manifest for this shard
                manifest_entry = {
                    "audio_filepath": str(shard_path) + f"/{audio_name}",
                    "duration": entry.get("duration", 0),
                    "text": entry.get("text", ""),
                    "tarred_audio_filepath": str(shard_path),
                    "tarred_audio_index": i,
                }
                manifest_entry.update(metadata)
                manifest_entries.append(manifest_entry)
        
        # Write shard manifest
        shard_manifest_path = self.output_dir / f"shard_{shard_idx:05d}_manifest.json"
        with open(shard_manifest_path, 'w') as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        logger.info(f"Wrote {len(manifest_entries)} entries to shard {shard_idx}")
        
        return str(shard_manifest_path)
    
    def process(self):
        """Process manifest and create WebDataset tar files."""
        logger.info(f"Starting WebDataset creation from {self.input_manifest_file}")
        
        # Load input manifest
        manifest_data = []
        with open(self.input_manifest_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    manifest_data.append(entry)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line.strip()}")
        
        logger.info(f"Loaded {len(manifest_data)} entries from manifest")
        
        # Group by session if maintaining integrity
        if self.maintain_session_integrity:
            sessions = self._group_by_session(manifest_data)
            logger.info(f"Grouped into {len(sessions)} sessions")
            
            # Create shards
            shards = self._create_shards(sessions)
        else:
            # Direct sharding without session grouping
            if self.shuffle_sessions:
                random.shuffle(manifest_data)
            
            # Split into shards
            shard_size = len(manifest_data) // self.num_shards + 1
            shards = [manifest_data[i:i+shard_size] 
                     for i in range(0, len(manifest_data), shard_size)]
        
        # Write shards
        shard_manifests = []
        for i, shard_data in enumerate(shards):
            if shard_data:  # Skip empty shards
                manifest_path = self._write_tar_shard(shard_data, i)
                shard_manifests.append(manifest_path)
        
        # Create combined manifest pointing to all shards
        combined_manifest = []
        for manifest_path in shard_manifests:
            with open(manifest_path, 'r') as f:
                for line in f:
                    combined_manifest.append(json.loads(line.strip()))
        
        # Write combined manifest
        write_manifest(self.output_manifest_file, combined_manifest)
        
        # Write shard list
        shard_list_path = self.output_dir / "shard_list.txt"
        with open(shard_list_path, 'w') as f:
            for i in range(len(shards)):
                if shards[i]:  # Only list non-empty shards
                    f.write(f"shard_{i:05d}.tar\n")
        
        logger.info(f"WebDataset creation complete:")
        logger.info(f"  - {len(shards)} tar shards created")
        logger.info(f"  - {len(combined_manifest)} total segments")
        logger.info(f"  - Output directory: {self.output_dir}")
        logger.info(f"  - Combined manifest: {self.output_manifest_file}")
        logger.info(f"  - Shard list: {shard_list_path}")