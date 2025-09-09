"""Simple file copy processor."""

import os
import shutil
from pathlib import Path
from sdp.processors.base_processor import BaseProcessor
from sdp.logging import logger


class CopyFile(BaseProcessor):
    """Copy a file from source to destination."""
    
    def __init__(
        self,
        input_file: str,
        output_file: str,
        **kwargs,
    ):
        """
        Initialize CopyFile processor.
        
        Args:
            input_file: Source file path
            output_file: Destination file path
        """
        # Extract any manifest-related kwargs to avoid conflicts
        kwargs.pop('input_manifest_file', None)
        kwargs.pop('output_manifest_file', None)
        
        # Use file paths as manifest paths for base class
        super().__init__(
            output_manifest_file=output_file,
            input_manifest_file=input_file,
            **kwargs
        )
        self.input_file = input_file
        self.output_file = output_file
    
    def process(self):
        """Copy the file from input to output location."""
        # Create output directory if it doesn't exist
        output_dir = Path(self.output_file).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        logger.info(f"Copying {self.input_file} to {self.output_file}")
        shutil.copy2(self.input_file, self.output_file)
        logger.info(f"File copied successfully")