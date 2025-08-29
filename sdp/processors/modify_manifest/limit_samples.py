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
from sdp.processors.base_processor import BaseProcessor
from sdp.logging import logger


class LimitManifestEntries(BaseProcessor):
    """
    Processor to limit the number of entries in a manifest file.
    
    This processor reads a manifest and outputs only the first N entries.
    If max_entries is -1, all entries are kept (no limiting).
    
    Args:
        max_entries (int): Maximum number of entries to keep. 
                          Use -1 to keep all entries (no limit).
        **kwargs: Additional arguments for BaseProcessor
    
    Example:
        # Keep only first 10 entries
        _target_: sdp.processors.modify_manifest.limit_samples.LimitManifestEntries
        max_entries: 10
        
        # Keep all entries (no limit)
        _target_: sdp.processors.modify_manifest.limit_samples.LimitManifestEntries
        max_entries: -1
    """
    
    def __init__(self, max_entries: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.max_entries = max_entries
    
    def process(self):
        """Process the manifest and limit entries if needed."""
        
        # Read input manifest
        entries = []
        with open(self.input_manifest_file, 'r') as f:
            for line in f:
                entries.append(json.loads(line.strip()))
        
        total_entries = len(entries)
        
        # Apply limiting if max_entries is not -1
        if self.max_entries > 0:
            entries = entries[:self.max_entries]
            logger.info(f"Limited manifest from {total_entries} to {len(entries)} entries")
        else:
            logger.info(f"Processing all {total_entries} entries (no limit applied)")
        
        # Write output manifest
        with open(self.output_manifest_file, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        logger.info(f"Output manifest written with {len(entries)} entries")