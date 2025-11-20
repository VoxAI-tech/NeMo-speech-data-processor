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


class CreateInitialManifestAudioBox(BaseProcessor):
    """Processor to create initial manifest for AudioBox drive-thru dataset.

    This processor creates a manifest from drive-thru order audio files where:
    - Each device has direct WAV files (no separate mic/spk)
    - The folder structure is: data/<device_id>/<year>/<month>/<day>/<hour>/<session_id>.wav

    Args:
        raw_data_dir (str): Path to the audiobox data folder
        brand_name (str): Full brand name for metadata (e.g., "Burger King")
        country_name (str): Full country name for metadata (e.g., "Germany")
        audio_type (str): Type of audio - typically "customer" for single-channel drive-thru

    Creates a manifest with:
        - audio_filepath: path to the audio file
        - text: placeholder text (needs to be transcribed)
        - session_id: unique identifier from the filename (without .wav)
        - device_id: device identifier from the path
        - timestamp: extracted from the path structure
        - brand: brand name
        - country: country name
        - dataset_tag: combined brand and country tag
        - audio_type: type of audio (customer/employee)
    """

    def __init__(
        self,
        raw_data_dir: str,
        brand_name: str = "Burger King",
        country_name: str = "Germany",
        audio_type: str = "customer",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.raw_data_dir = raw_data_dir
        self.brand_name = brand_name
        self.country_name = country_name
        self.audio_type = audio_type

    def process(self):
        """Process audio files and create manifest"""
        entries = []
        dataset_tag = f"{self.brand_name} {self.country_name}"

        if not os.path.exists(self.raw_data_dir):
            raise FileNotFoundError(f"Directory not found: {self.raw_data_dir}")

        print(f"Processing {dataset_tag} data from: {self.raw_data_dir}")
        print(f"Audio type: {self.audio_type}")

        # Walk through all subdirectories to find WAV files
        for root, dirs, files in os.walk(self.raw_data_dir):
            for file in files:
                if file.endswith('.wav'):
                    audio_file = os.path.join(root, file)

                    # Extract metadata from path
                    path_parts = Path(root).parts

                    try:
                        # Expected structure: .../device_id/year/month/day/hour/
                        # Find the audiobox folder index
                        audiobox_idx = -1
                        for i, part in enumerate(path_parts):
                            if 'audiobox' in part:
                                audiobox_idx = i
                                break

                        if audiobox_idx >= 0 and audiobox_idx + 5 < len(path_parts):
                            device_id = path_parts[audiobox_idx + 1]
                            year = path_parts[audiobox_idx + 2]
                            month = path_parts[audiobox_idx + 3]
                            day = path_parts[audiobox_idx + 4]
                            hour = path_parts[audiobox_idx + 5]
                            session_id = Path(file).stem  # filename without extension

                            entry = {
                                "audio_filepath": os.path.abspath(audio_file),
                                "text": "",  # Empty text - needs transcription
                                "session_id": session_id,
                                "device_id": device_id,
                                "timestamp": f"{year}-{month.zfill(2)}-{day.zfill(2)}T{hour.zfill(2)}:00:00",
                                "brand": self.brand_name,
                                "country": self.country_name,
                                "dataset_tag": dataset_tag,
                                "audio_type": self.audio_type
                            }
                            entries.append(entry)
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Could not parse path structure for {audio_file}: {e}")
                        continue

        # Sort entries by audio filepath for consistency
        entries.sort(key=lambda x: x["audio_filepath"])

        print(f"Found {len(entries)} WAV files")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_manifest_file), exist_ok=True)

        # Write manifest
        with open(self.output_manifest_file, "w") as fout:
            for entry in entries:
                fout.write(json.dumps(entry) + "\n")

        print(f"Manifest created: {self.output_manifest_file}")
