"""
Processor to convert audio files while preserving folder structure with session IDs
"""
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, List
from sdp.processors.base_processor import BaseParallelProcessor
from sdp.processors.base_processor import DataEntry


class FfmpegConvertPreserveStructure(BaseParallelProcessor):
    """
    Converts audio files to target format while preserving the session ID folder structure.
    Extracts the session ID (UUID) from the original path and maintains it in the output.
    """
    
    def __init__(
        self,
        output_manifest_file: str,
        converted_audio_dir: str,
        input_file_key: str = "source_audio_filepath",
        output_file_key: str = "audio_filepath",
        target_samplerate: int = 16000,
        target_nchannels: int = 1,
        target_format: str = "flac",
        **kwargs,
    ):
        super().__init__(output_manifest_file=output_manifest_file, **kwargs)
        self.converted_audio_dir = converted_audio_dir
        self.input_file_key = input_file_key
        self.output_file_key = output_file_key
        self.target_samplerate = target_samplerate
        self.target_nchannels = target_nchannels
        self.target_format = target_format
        
        # Create output directory if it doesn't exist
        os.makedirs(self.converted_audio_dir, exist_ok=True)

    def process_dataset_entry(self, data_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single dataset entry, converting the audio file while preserving structure.
        """
        if self.input_file_key not in data_entry:
            return None
            
        input_filepath = data_entry[self.input_file_key]
        
        # Extract session ID (UUID) from the path
        # Path structure: .../data/ej/au/DEVICE_ID/YYYY/MM/DD/HH/SESSION_UUID/mic.wav
        path_parts = Path(input_filepath).parts
        
        # Find the UUID part (it's the second-to-last part, before mic.wav)
        if len(path_parts) >= 2:
            session_id = path_parts[-2]  # The UUID folder
            
            # Find device_id (should be 4 parts before the UUID)
            if len(path_parts) >= 9:
                # Get device_id and date parts for better organization
                device_id = path_parts[-9]  # e.g., '90104C41'
                year = path_parts[-8]
                month = path_parts[-7] 
                day = path_parts[-6]
                hour = path_parts[-5]
                
                # Create a structured output path
                output_filename = f"{device_id}_{year}{month}{day}_{hour}_{session_id}.{self.target_format}"
            else:
                # Fallback to just session_id
                output_filename = f"{session_id}.{self.target_format}"
        else:
            # Fallback to original filename with extension changed
            base_name = Path(input_filepath).stem
            output_filename = f"{base_name}.{self.target_format}"
            
        output_filepath = os.path.join(self.converted_audio_dir, output_filename)
        
        # Check if file already exists
        if os.path.exists(output_filepath):
            data_entry[self.output_file_key] = output_filepath
            return [DataEntry(data=data_entry)]
            
        # Build ffmpeg command
        ffmpeg_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-i", input_filepath,
            "-ar", str(self.target_samplerate),
            "-ac", str(self.target_nchannels),
            "-y",  # Overwrite output files
            output_filepath
        ]
        
        # Run the conversion
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg error for {input_filepath}: {result.stderr}")
                return None
            data_entry[self.output_file_key] = output_filepath
            return [DataEntry(data=data_entry)]
        except Exception as e:
            print(f"Error converting {input_filepath}: {e}")
            return None