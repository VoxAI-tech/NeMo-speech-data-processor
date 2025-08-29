#!/usr/bin/env python3
"""
Initial Manifest Generator for EJ AU Drive-thru Dataset
Creates manifest files from hierarchical audio directory structure
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class InitialManifestGenerator:
    """
    Generate initial manifest from hierarchical audio directory structure.
    
    Directory structure expected:
    data/audio/{country}/{location}/{device_id}/{year}/{month}/{day}/{hour}/{session_id}/{mic.wav|spk.wav}
    
    Example:
    data/audio/ej/au/90104C41/2024/12/30/08/5c3b5675-80b4-4605-a716-f425358f7e4a/mic.wav
    """
    
    def __init__(self, 
                 data_dir: str,
                 output_manifest: str,
                 audio_type: str = "both",
                 preserve_structure: bool = True):
        """
        Initialize the manifest generator.
        
        Args:
            data_dir: Root directory containing audio files
            output_manifest: Path to output manifest JSON file
            audio_type: Type of audio to include ("mic", "spk", "both")
            preserve_structure: Whether to preserve directory structure in metadata
        """
        self.data_dir = Path(data_dir)
        self.output_manifest = Path(output_manifest)
        self.audio_type = audio_type
        self.preserve_structure = preserve_structure
        self.entries = []
        
    def parse_audio_path(self, audio_path: Path) -> Optional[Dict]:
        """
        Parse audio file path to extract metadata.
        
        Returns dict with:
        - source_audio_filepath: Full path to audio file
        - country: Country code (e.g., "ej")
        - location: Location code (e.g., "au")
        - device_id: Device identifier
        - timestamp: ISO format timestamp
        - session_id: Session UUID
        - channel: Audio channel ("mic" or "spk")
        """
        try:
            # Get relative path from data_dir
            rel_path = audio_path.relative_to(self.data_dir)
            parts = rel_path.parts
            
            # Expected structure: country/location/device_id/year/month/day/hour/session_id/filename
            # (without 'audio' prefix since data_dir already points to data/audio)
            if len(parts) != 9:
                logger.warning(f"Skipping file with unexpected structure: {audio_path}")
                return None
            
            # Extract components (0-indexed)
            country = parts[0]
            location = parts[1]
            device_id = parts[2]
            year = parts[3]
            month = parts[4]
            day = parts[5]
            hour = parts[6]
            session_id = parts[7]
            filename = parts[8]
            
            # Determine channel from filename
            if filename == "mic.wav":
                channel = "mic"
            elif filename == "spk.wav":
                channel = "spk"
            else:
                logger.warning(f"Unknown audio file type: {filename}")
                return None
            
            # Check if we should include this channel
            if self.audio_type != "both" and channel != self.audio_type:
                return None
            
            # Create timestamp
            timestamp = f"{year}-{month}-{day}T{hour}:00:00"
            
            # Build metadata entry
            entry = {
                "source_audio_filepath": str(audio_path.absolute()),
                "country": country,
                "location": location,
                "device_id": device_id,
                "timestamp": timestamp,
                "session_id": session_id,
                "channel": channel,
                "year": year,
                "month": month,
                "day": day,
                "hour": hour
            }
            
            # Add hierarchical path if preserving structure
            if self.preserve_structure:
                entry["relative_path"] = str(rel_path)
                entry["hierarchy"] = f"{country}/{location}/{device_id}/{year}/{month}/{day}/{hour}/{session_id}"
            
            return entry
            
        except Exception as e:
            logger.error(f"Error parsing path {audio_path}: {e}")
            return None
    
    def scan_directory(self) -> List[Dict]:
        """
        Scan directory for audio files and generate manifest entries.
        """
        logger.info(f"Scanning directory: {self.data_dir}")
        
        # Find all WAV files
        wav_files = list(self.data_dir.rglob("*.wav"))
        logger.info(f"Found {len(wav_files)} WAV files")
        
        # Process each file
        for wav_file in wav_files:
            entry = self.parse_audio_path(wav_file)
            if entry:
                self.entries.append(entry)
        
        logger.info(f"Generated {len(self.entries)} manifest entries")
        
        # Sort entries for consistency
        self.entries.sort(key=lambda x: (
            x.get("device_id", ""),
            x.get("timestamp", ""),
            x.get("session_id", ""),
            x.get("channel", "")
        ))
        
        return self.entries
    
    def write_manifest(self, entries: Optional[List[Dict]] = None) -> None:
        """
        Write manifest entries to output file.
        """
        if entries is None:
            entries = self.entries
        
        # Create output directory if needed
        self.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        
        # Write manifest as JSONL
        with open(self.output_manifest, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Wrote {len(entries)} entries to {self.output_manifest}")
    
    def generate_statistics(self) -> Dict:
        """
        Generate statistics about the dataset.
        """
        stats = {
            "total_files": len(self.entries),
            "devices": set(),
            "sessions": set(),
            "channels": {"mic": 0, "spk": 0},
            "dates": set()
        }
        
        for entry in self.entries:
            stats["devices"].add(entry.get("device_id", ""))
            stats["sessions"].add(entry.get("session_id", ""))
            stats["channels"][entry.get("channel", "")] += 1
            date = f"{entry.get('year', '')}-{entry.get('month', '')}-{entry.get('day', '')}"
            stats["dates"].add(date)
        
        # Convert sets to counts
        stats["num_devices"] = len(stats["devices"])
        stats["num_sessions"] = len(stats["sessions"])
        stats["num_dates"] = len(stats["dates"])
        
        # Remove sets for JSON serialization
        del stats["devices"]
        del stats["sessions"]
        del stats["dates"]
        
        return stats
    
    def generate(self) -> None:
        """
        Main generation method.
        """
        # Scan directory
        self.scan_directory()
        
        # Generate statistics
        stats = self.generate_statistics()
        logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")
        
        # Write manifest
        self.write_manifest()
        
        # Write statistics file
        stats_file = self.output_manifest.with_suffix('.stats.json')
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Wrote statistics to {stats_file}")


def main():
    """Command-line interface for manifest generation."""
    parser = argparse.ArgumentParser(
        description="Generate initial manifest for EJ AU drive-thru dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory containing audio files"
    )
    parser.add_argument(
        "--output-manifest",
        type=str,
        required=True,
        help="Path to output manifest JSON file"
    )
    parser.add_argument(
        "--audio-type",
        type=str,
        choices=["mic", "spk", "both"],
        default="both",
        help="Type of audio to include (default: both)"
    )
    parser.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Preserve directory structure in metadata"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = InitialManifestGenerator(
        data_dir=args.data_dir,
        output_manifest=args.output_manifest,
        audio_type=args.audio_type,
        preserve_structure=args.preserve_structure
    )
    
    # Generate manifest
    generator.generate()


if __name__ == "__main__":
    main()