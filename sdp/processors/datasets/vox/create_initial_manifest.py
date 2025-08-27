# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from pathlib import Path

from sdp.processors.base_processor import BaseProcessor


class CreateInitialManifestVox(BaseProcessor):
    """Processor to create initial manifest for drive-thru VOX dataset.
    
    This processor creates a manifest from drive-thru order audio files where:
    - Each session is in a folder with mic.wav (customer) and spk.wav (employee)
    - You can process either mic.wav or spk.wav files based on configuration
    - The folder structure is: data/<brand_code>/<country_code>/<device_id>/<year>/<month>/<day>/<hour>/<session_id>/<audio_type>.wav
    
    Args:
        raw_data_dir (str): Path to the data folder
        brand_code (str): Brand code (e.g., "ej" for El Jannah, "bk" for Burger King)
        country_code (str): Country code (e.g., "au" for Australia, "pl" for Poland)
        brand_name (str): Full brand name for metadata (e.g., "El Jannah")
        country_name (str): Full country name for metadata (e.g., "Australia")
        audio_type (str): Type of audio to process - "mic" for customer or "spk" for employee
        
    Creates a manifest with:
        - audio_filepath: path to the audio file (mic.wav or spk.wav)
        - text: placeholder text (needs to be transcribed)
        - session_id: unique identifier from the session folder
        - device_id: device identifier from the path
        - timestamp: extracted from the path structure
        - brand: brand name
        - country: country name
        - dataset_tag: combined brand and country tag
        - audio_type: "customer" or "employee" based on mic/spk selection
    """
    
    def __init__(
        self,
        raw_data_dir: str,
        brand_code: str = "ej",
        country_code: str = "au",
        brand_name: str = "El Jannah",
        country_name: str = "Australia",
        audio_type: str = "mic",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = raw_data_dir
        self.brand_code = brand_code
        self.country_code = country_code
        self.brand_name = brand_name
        self.country_name = country_name
        self.audio_type = audio_type
        
        if self.audio_type not in ["mic", "spk"]:
            raise ValueError(f"audio_type must be 'mic' or 'spk', got '{self.audio_type}'")
    
    def process(self):
        """Process audio files and create manifest"""
        entries = []
        dataset_tag = f"{self.brand_name} {self.country_name}"
        audio_filename = f"{self.audio_type}.wav"
        audio_role = "customer" if self.audio_type == "mic" else "employee"
        
        # Find the brand/country data directory
        data_dir = os.path.join(self.raw_data_dir, self.brand_code, self.country_code)
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory not found: {data_dir}")
        
        print(f"Processing {dataset_tag} data from: {data_dir}")
        print(f"Audio type: {self.audio_type} ({audio_role})")
        
        # Walk through all subdirectories to find audio files
        for root, dirs, files in os.walk(data_dir):
            if audio_filename in files:
                audio_file = os.path.join(root, audio_filename)
                
                # Extract metadata from path
                path_parts = Path(root).parts
                
                # Find indices of relevant parts
                try:
                    country_idx = path_parts.index(self.country_code)
                    if country_idx + 6 < len(path_parts):
                        device_id = path_parts[country_idx + 1]
                        year = path_parts[country_idx + 2]
                        month = path_parts[country_idx + 3]
                        day = path_parts[country_idx + 4]
                        hour = path_parts[country_idx + 5]
                        session_id = path_parts[country_idx + 6]
                        
                        entry = {
                            "audio_filepath": os.path.abspath(audio_file),
                            "text": "",  # Empty text - needs transcription
                            "session_id": session_id,
                            "device_id": device_id,
                            "timestamp": f"{year}-{month}-{day}T{hour}:00:00",
                            "brand": self.brand_name,
                            "country": self.country_name,
                            "dataset_tag": dataset_tag,
                            "audio_type": audio_role
                        }
                        entries.append(entry)
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse path structure for {audio_file}")
                    continue
        
        # Sort entries by audio filepath for consistency
        entries.sort(key=lambda x: x["audio_filepath"])
        
        print(f"Found {len(entries)} {audio_filename} files")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)
        
        # Write manifest
        with open(self.output_manifest_file, "w") as fout:
            for entry in entries:
                fout.write(json.dumps(entry) + "\n")
        
        print(f"Manifest created: {self.output_manifest_file}")