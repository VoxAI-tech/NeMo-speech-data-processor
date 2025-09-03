#!/usr/bin/env python
"""
ConvertToWebDataset processor - Creates WebDataset TAR archives with wav/json pairs.
Compatible with HuggingFace datasets and webdataset library for efficient streaming.
"""

import json
import os
import tarfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import math

import soundfile as sf
import torch
import torchaudio
from sdp.processors.base_processor import BaseProcessor
from tqdm import tqdm


class ConvertToWebDataset(BaseProcessor):
    """
    Converts NeMo manifest format to WebDataset TAR archives with wav/json pairs.
    
    WebDataset format:
    - Each TAR archive contains multiple samples
    - Each sample consists of files with the same prefix (e.g., sample_0001.wav, sample_0001.json)
    - TAR archives are sharded for efficient streaming
    - Compatible with HuggingFace datasets and webdataset library
    
    Args:
        output_dir: Output directory for TAR files
        shard_size: Target size for each shard in MB (default: 1000 MB = 1GB)
        num_shards: Fixed number of shards (overrides shard_size if set)
        split: Dataset split name (default: "train")
        prefix: Prefix for shard filenames (default: "shard")
        shuffle: Whether to shuffle samples before creating shards
        slice_with_offset: If True, extract audio segments using offset/duration
        audio_type_field: Field name for audio type classification
        include_metadata: If True, include all manifest fields in JSON output
    """
    
    def __init__(
        self,
        output_manifest_file: str,
        input_manifest_file: str = None,
        output_dir: str = None,
        shard_size: int = 1000,  # MB per shard
        num_shards: Optional[int] = None,
        split: str = "train",
        prefix: str = "shard",
        shuffle: bool = True,
        shuffle_seed: int = 42,
        slice_with_offset: bool = True,
        audio_type_field: str = "audio_type",
        include_metadata: bool = True,
        **kwargs,
    ):
        super().__init__(
            input_manifest_file=input_manifest_file,
            output_manifest_file=output_manifest_file
        )
        # If output_dir not specified, derive from output_manifest_file
        if output_dir is None:
            output_dir = Path(output_manifest_file).parent / "webdataset"
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size * 1024 * 1024  # Convert MB to bytes
        self.num_shards = num_shards
        self.split = split
        self.prefix = prefix
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.slice_with_offset = slice_with_offset
        self.audio_type_field = audio_type_field
        self.include_metadata = include_metadata
        
    def process(self):
        """Process manifest entries and create WebDataset TAR archives."""
        
        # Create output directories
        split_dir = self.output_dir / self.split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Load the input manifest
        manifest_entries = []
        with open(self.input_manifest_file, 'r') as f:
            for line in f:
                if line.strip():
                    manifest_entries.append(json.loads(line))
        
        if not manifest_entries:
            print("No entries found in manifest")
            return
            
        # Shuffle if requested
        if self.shuffle:
            import random
            random.seed(self.shuffle_seed)
            random.shuffle(manifest_entries)
        
        # Calculate sharding
        if self.num_shards:
            # Fixed number of shards
            samples_per_shard = math.ceil(len(manifest_entries) / self.num_shards)
            shards = [manifest_entries[i:i+samples_per_shard] 
                     for i in range(0, len(manifest_entries), samples_per_shard)]
        else:
            # Dynamic sharding based on size
            shards = self._create_shards_by_size(manifest_entries)
        
        print(f"Creating {len(shards)} WebDataset shards in {split_dir}")
        
        # Track statistics
        total_duration = 0
        total_samples = 0
        audio_types_count = {}
        shard_info = []
        
        # Process each shard
        for shard_idx, shard_entries in enumerate(tqdm(shards, desc="Creating shards")):
            shard_filename = f"{self.prefix}_{shard_idx:06d}.tar"
            shard_path = split_dir / shard_filename
            
            shard_stats = self._create_shard(shard_entries, shard_path, shard_idx)
            shard_info.append({
                "shard_id": shard_idx,
                "filename": shard_filename,
                "num_samples": shard_stats["num_samples"],
                "duration_seconds": shard_stats["duration"],
                "size_mb": shard_stats["size_bytes"] / (1024 * 1024)
            })
            
            total_duration += shard_stats["duration"]
            total_samples += shard_stats["num_samples"]
            
            # Update audio type counts
            for audio_type, count in shard_stats["audio_types"].items():
                audio_types_count[audio_type] = audio_types_count.get(audio_type, 0) + count
        
        # Write output manifest with shard information
        output_entries = []
        for shard in shard_info:
            output_entries.append({
                "tar_filepath": str(split_dir / shard["filename"]),
                "num_samples": shard["num_samples"],
                "duration_seconds": shard["duration_seconds"],
                "size_mb": shard["size_mb"],
                "shard_id": shard["shard_id"]
            })
        
        with open(self.output_manifest_file, 'w') as f:
            for entry in output_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        
        # Write dataset metadata
        metadata = {
            "format": "webdataset",
            "split": self.split,
            "total_shards": len(shards),
            "total_samples": total_samples,
            "total_duration_seconds": total_duration,
            "total_duration_hours": total_duration / 3600,
            "audio_types": audio_types_count,
            "shard_size_mb": self.shard_size / (1024 * 1024),
            "shards": shard_info,
            "slice_with_offset": self.slice_with_offset,
            "shuffled": self.shuffle
        }
        
        metadata_path = self.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nWebDataset creation complete:")
        print(f"  Total shards: {len(shards)}")
        print(f"  Total samples: {total_samples}")
        print(f"  Total duration: {total_duration / 3600:.2f} hours")
        print(f"  Audio types: {audio_types_count}")
        print(f"  Output: {split_dir}")
        print(f"  Metadata: {metadata_path}")
        
    def _create_shards_by_size(self, entries: List[Dict]) -> List[List[Dict]]:
        """Create shards based on target size."""
        shards = []
        current_shard = []
        current_size = 0
        
        for entry in entries:
            # Estimate size (duration * sample_rate * bytes_per_sample)
            duration = entry.get("duration", 0)
            estimated_size = duration * 16000 * 2  # 16kHz, 16-bit
            
            if current_shard and (current_size + estimated_size) > self.shard_size:
                shards.append(current_shard)
                current_shard = []
                current_size = 0
            
            current_shard.append(entry)
            current_size += estimated_size
        
        if current_shard:
            shards.append(current_shard)
        
        return shards
    
    def _create_shard(self, entries: List[Dict], shard_path: Path, shard_idx: int) -> Dict:
        """Create a single TAR shard with wav/json pairs."""
        stats = {
            "num_samples": 0,
            "duration": 0,
            "audio_types": {},
            "size_bytes": 0
        }
        
        # Create temporary directory for files
        temp_dir = Path(f"/tmp/webdataset_temp_{shard_idx}")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Process each entry
            for idx, entry in enumerate(entries):
                # Create sample key using session_id and segment_id format
                # Format: {session_id}_segment_{segment_id:04d}
                session_id = entry.get("session_id", f"unknown_{idx:08d}")
                segment_id = entry.get("segment_id", idx)
                
                # Ensure segment_id is properly formatted as 4-digit number
                if isinstance(segment_id, str):
                    try:
                        segment_id = int(segment_id)
                    except:
                        segment_id = idx
                
                sample_key = f"{session_id}_segment_{segment_id:04d}"
                
                # Extract or copy audio
                wav_filename = f"{sample_key}.wav"
                wav_path = temp_dir / wav_filename
                
                if self.slice_with_offset and "source_audio_offset" in entry and "duration" in entry:
                    # Extract segment from source audio
                    source_audio = entry["audio_filepath"]
                    offset = entry["source_audio_offset"]
                    duration = entry["duration"]
                    
                    try:
                        # Load and slice audio
                        waveform, sample_rate = torchaudio.load(
                            source_audio,
                            frame_offset=int(offset * 16000),  # Assuming 16kHz
                            num_frames=int(duration * 16000)
                        )
                        
                        # Save sliced audio
                        torchaudio.save(str(wav_path), waveform, sample_rate)
                        
                    except Exception as e:
                        print(f"Error extracting audio segment: {e}")
                        # Try to copy the whole file as fallback
                        if os.path.exists(source_audio):
                            shutil.copy2(source_audio, wav_path)
                        else:
                            continue
                else:
                    # Copy the audio file
                    source_audio = entry["audio_filepath"]
                    if os.path.exists(source_audio):
                        shutil.copy2(source_audio, wav_path)
                    else:
                        print(f"Source audio not found: {source_audio}")
                        continue
                
                # Create JSON metadata with matching filename
                json_filename = f"{sample_key}.json"
                json_path = temp_dir / json_filename
                
                json_data = self._create_json_metadata(entry)
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, ensure_ascii=False)
                
                # Update statistics
                stats["num_samples"] += 1
                stats["duration"] += entry.get("duration", 0)
                
                audio_type = entry.get(self.audio_type_field, "unknown")
                stats["audio_types"][audio_type] = stats["audio_types"].get(audio_type, 0) + 1
            
            # Create TAR archive
            with tarfile.open(shard_path, 'w') as tar:
                for file_path in sorted(temp_dir.iterdir()):
                    tar.add(file_path, arcname=file_path.name)
            
            stats["size_bytes"] = shard_path.stat().st_size
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return stats
    
    def _create_json_metadata(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create JSON metadata in WebDataset format.
        
        Args:
            entry: Manifest entry
            
        Returns:
            JSON metadata dictionary
        """
        # Core fields for WebDataset
        json_data = {
            "text": entry.get("text", ""),
            "transcription": entry.get("text", ""),  # Duplicate for compatibility
        }
        
        # Add segment information
        if "segment_id" in entry:
            json_data["segment_id"] = entry["segment_id"]
            
        # Add timing information
        if "duration" in entry:
            json_data["duration"] = entry["duration"]
        if "source_audio_offset" in entry:
            json_data["offset"] = entry["source_audio_offset"]
            
        # Add session/device information
        if "session_id" in entry:
            json_data["session_id"] = entry["session_id"]
        if "device_id" in entry:
            json_data["device_id"] = entry["device_id"]
            
        # Add audio type classification
        if self.audio_type_field in entry:
            json_data["audio_type"] = entry[self.audio_type_field]
            json_data["speaker"] = entry[self.audio_type_field]  # Alternative key
            
        # Add quality/confidence metrics if available
        if "cross_channel_confidence" in entry:
            json_data["confidence"] = entry["cross_channel_confidence"]
        if "audio_bleeding_detected" in entry:
            json_data["audio_bleeding"] = entry["audio_bleeding_detected"]
        if "correction_source" in entry:
            json_data["correction_source"] = entry["correction_source"]
            
        # Add language information
        if "language" in entry:
            json_data["language"] = entry["language"]
            
        # Include all other metadata if requested
        if self.include_metadata:
            # Add remaining fields that aren't already included
            for key, value in entry.items():
                if key not in json_data and key not in ["audio_filepath", "json_filepath"]:
                    json_data[key] = value
                    
        return json_data