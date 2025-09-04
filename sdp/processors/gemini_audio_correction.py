"""
Gemini Audio Correction Processor
Listens to audio segments and provides corrected transcriptions focusing on menu item spellings.
"""

import json
import pathlib
import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from datetime import datetime, timedelta

from sdp.processors.base_processor import BaseParallelProcessor
from sdp.logging import logger

try:
    from google import genai
    from google.genai import types
    from google.genai.types import GenerateContentConfig
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Google GenAI SDK not installed. Install with: uv pip install google-genai")


@dataclass
class GeminiConfig:
    """Configuration for Gemini API"""
    api_key: str
    model: str = "gemini-2.5-pro"  # or "gemini-1.5-pro" for better accuracy
    temperature: float = 0.1  # Very low for consistent corrections
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30


class GeminiAudioCorrection(BaseParallelProcessor):
    """
    Processor that uses Google Gemini to listen to audio segments and correct transcriptions.
    Focuses strictly on correcting menu item spellings while preserving everything else verbatim.
    """
    
    def __init__(
        self,
        input_manifest_file: str,
        output_manifest_file: str,
        audio_filepath_field: str = "audio_filepath",
        original_text_field: str = "text",
        corrected_text_field: str = "gemini_corrected_text",
        confidence_field: str = "gemini_confidence",
        correction_made_field: str = "gemini_correction_made",
        offset_field: str = "offset",
        duration_field: str = "duration",
        prompt_file: Optional[str] = None,  # Path to YAML prompt file
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.1,
        batch_size: int = 1,  # Process one at a time for audio
        max_workers: int = 10,  # Number of parallel workers
        requests_per_minute: int = 900,  # Stay under 1000 RPM limit
        save_errors: bool = True,
        error_log_file: Optional[str] = None,
        **kwargs
    ):
        super().__init__(
            input_manifest_file=input_manifest_file,
            output_manifest_file=output_manifest_file,
            **kwargs
        )
        
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Gemini SDK is required. Install with: pip install google-generativeai"
            )
        
        self.audio_filepath_field = audio_filepath_field
        self.original_text_field = original_text_field
        self.corrected_text_field = corrected_text_field
        self.confidence_field = confidence_field
        self.correction_made_field = correction_made_field
        self.offset_field = offset_field
        self.duration_field = duration_field
        self.batch_size = batch_size
        self.save_errors = save_errors
        self.prompt_file = prompt_file
        self.max_workers = max_workers
        self.requests_per_minute = requests_per_minute
        
        # Rate limiting setup
        self.rate_limiter = Semaphore(max_workers)  # Limit concurrent requests
        self.request_times = []  # Track request times for RPM limiting
        self.rpm_lock = Semaphore(1)  # Thread-safe access to request_times
        
        # Load prompt template if provided
        self.prompt_template = None
        if self.prompt_file:
            with open(self.prompt_file, 'r') as f:
                self.prompt_template = yaml.safe_load(f)
        
        # Set up error logging
        if error_log_file:
            self.error_log_file = error_log_file
        else:
            output_dir = os.path.dirname(output_manifest_file)
            self.error_log_file = os.path.join(output_dir, "gemini_errors.json")
        
        # Initialize Gemini client
        self.config = GeminiConfig(
            api_key=api_key or os.environ.get("GEMINI_API_KEY"),
            model=model,
            temperature=temperature
        )
        
        if not self.config.api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter"
            )
        
        # Initialize GenAI client
        self.client = genai.Client(api_key=self.config.api_key)
        
        # Statistics (with thread-safe access)
        self.stats = {
            "total_processed": 0,
            "corrections_made": 0,
            "errors": 0,
            "api_calls": 0
        }
        self.stats_lock = Semaphore(1)  # Thread-safe access to stats
    
    def build_correction_prompt(self, original_transcription: str) -> str:
        """
        Build the prompt for Gemini with menu vocabulary and strict instructions.
        Uses external prompt template if provided, otherwise uses default.
        """
        if self.prompt_template:
            # Use external prompt template
            system_prompt = self.prompt_template.get('system', '')
            user_prompt = self.prompt_template.get('user', '')
            
            # Replace placeholders
            user_prompt = user_prompt.replace('{original_text}', original_transcription)
            
            # Combine system and user prompts
            prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Add JSON format instruction
            prompt += """\n\nReturn a JSON object with exactly this structure:
{{
  "corrected_text": "the corrected transcription keeping ALL filler words (uh, um, etc.)",
  "confidence": 0.95,
  "corrections_made": ["list of specific menu item corrections if any"]
}}

If no corrections are needed, return the original text with an empty corrections_made list."""
        else:
            # Use default prompt with emphasis on preserving filler words
            prompt = f"""You are a transcription corrector for El Jannah Australia drive-thru orders.
Listen to the audio and compare it with the provided transcription.
Your ONLY task is to correct misspellings of menu items. Keep EVERYTHING else EXACTLY as it is.

CRITICAL: PRESERVE ALL FILLER WORDS (uh, um, ah, er, okay, like, you know, well, etc.)
Do NOT remove any words. Transcribe VERBATIM - exactly as spoken.

ORIGINAL TRANSCRIPTION:
"{original_transcription}"

MENU VOCABULARY FOR REFERENCE:
Chicken: 1/2 chicken, 1/4 chicken, whole chicken, chicken roll, chicken burger, wings, tenders
Sauces: garlic sauce, chilli sauce, tahini, mayo, bbq sauce, tomato sauce, ej sauce
Sides: chips, tabouli, coleslaw, pickles, hommous, babaghanouj, fattoush salad
Drinks: pepsi, pepsi max, mountain dew, 7 up, red bull, lipton ice tea
Other: falafel, roll pack

COMMON CORRECTIONS (only if heard in audio):
- tubuli/tabuli/tabooly → tabouli
- falafal/fallafel → falafel
- hommus/humus → hommous
- cole slaw → coleslaw
- chili → chilli
- garlick → garlic
- fatoosh → fattoush

CRITICAL RULES:
1. ONLY correct spelling of menu items based on what you hear
2. Do NOT change punctuation, capitalization, grammar, or word order
3. Do NOT add or remove ANY words - KEEP ALL FILLER WORDS
4. If someone says "uh" or "um", KEEP IT in the transcription
5. Only make a correction if you clearly hear a different menu item than written
6. Preserve natural speech patterns exactly as spoken

Return a JSON object with exactly this structure:
{{
  "corrected_text": "the corrected transcription with ALL filler words preserved",
  "confidence": 0.95,
  "corrections_made": ["list of specific menu corrections if any"]
}}

If no corrections are needed, return the original text with an empty corrections_made list."""
        
        return prompt
    
    def extract_audio_segment(self, audio_path: str, offset: float, duration: float, temp_dir: str) -> pathlib.Path:
        """
        Extract a segment of audio using ffmpeg.
        """
        import subprocess
        import tempfile
        
        # Create temp file for segment
        segment_file = pathlib.Path(temp_dir) / f"segment_{offset}_{duration}.wav"
        
        # Use ffmpeg to extract segment
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(offset),
            "-t", str(duration),
            "-i", audio_path,
            "-ar", "16000",  # 16kHz sample rate
            "-ac", "1",      # Mono
            "-c:a", "pcm_s16le",  # 16-bit PCM
            str(segment_file)
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return segment_file
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise
    
    def wait_for_rate_limit(self):
        """
        Ensure we don't exceed the rate limit (RPM).
        """
        with self.rpm_lock:
            now = datetime.now()
            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if now - t < timedelta(minutes=1)]
            
            # If we're at the limit, wait
            if len(self.request_times) >= self.requests_per_minute:
                # Calculate how long to wait
                oldest_request = self.request_times[0]
                wait_time = (oldest_request + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
                    # Clean up old requests again
                    now = datetime.now()
                    self.request_times = [t for t in self.request_times if now - t < timedelta(minutes=1)]
            
            # Add this request time
            self.request_times.append(now)
    
    def transcribe_with_gemini(self, audio_path: pathlib.Path, original_text: str) -> Dict[str, Any]:
        """
        Send audio to Gemini for correction with rate limiting.
        """
        # Wait for rate limit if needed
        self.wait_for_rate_limit()
        
        prompt = self.build_correction_prompt(original_text)
        
        try:
            # Read audio file
            audio_data = audio_path.read_bytes()
            audio_part = types.Part.from_bytes(
                data=audio_data, 
                mime_type="audio/wav"
            )
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=[prompt, audio_part],
                config=GenerateContentConfig(
                    response_modalities=["TEXT"],
                    response_mime_type="application/json",
                    temperature=self.config.temperature
                ),
            )
            
            with self.stats_lock:
                self.stats["api_calls"] += 1
            
            # Parse JSON response
            try:
                result = json.loads(response.text)
                
                # Validate response structure
                if not isinstance(result, dict):
                    raise ValueError("Response is not a JSON object")
                
                if "corrected_text" not in result:
                    result["corrected_text"] = original_text
                
                if "confidence" not in result:
                    result["confidence"] = 0.5
                
                if "corrections_made" not in result:
                    result["corrections_made"] = []
                
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Gemini response as JSON: {e}")
                # Return original text if parsing fails
                return {
                    "corrected_text": original_text,
                    "confidence": 0.0,
                    "corrections_made": [],
                    "error": f"JSON parse error: {str(e)}"
                }
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {
                "corrected_text": original_text,
                "confidence": 0.0,
                "corrections_made": [],
                "error": str(e)
            }
    
    def process_dataset_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single manifest entry.
        Required method for BaseParallelProcessor.
        """
        import tempfile
        
        with self.stats_lock:
            self.stats["total_processed"] += 1
        
        # Get required fields
        audio_filepath = entry.get(self.audio_filepath_field)
        original_text = entry.get(self.original_text_field, "")
        offset = entry.get(self.offset_field, 0)
        duration = entry.get(self.duration_field, 0)
        
        if not audio_filepath or not os.path.exists(audio_filepath):
            logger.warning(f"Audio file not found: {audio_filepath}")
            entry[self.corrected_text_field] = original_text
            entry[self.confidence_field] = 0.0
            entry[self.correction_made_field] = False
            with self.stats_lock:
                self.stats["errors"] += 1
            return entry
        
        # Extract audio segment
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Extract segment
                segment_path = self.extract_audio_segment(
                    audio_filepath, offset, duration, temp_dir
                )
                # Get correction from Gemini
                result = self.transcribe_with_gemini(segment_path, original_text)
                
                # Update entry
                corrected_text = result.get("corrected_text", original_text)
                entry[self.corrected_text_field] = corrected_text
                entry[self.confidence_field] = result.get("confidence", 0.0)
                entry[self.correction_made_field] = (corrected_text != original_text)
                
                # Add correction details if any
                if result.get("corrections_made"):
                    entry["gemini_corrections"] = result["corrections_made"]
                
                if entry[self.correction_made_field]:
                    with self.stats_lock:
                        self.stats["corrections_made"] += 1
                    logger.info(
                        f"Correction made - Segment {entry.get('segment_id', 'unknown')}: "
                        f"'{original_text[:50]}...' → '{corrected_text[:50]}...'"
                    )
                
                # Log errors if any
                if "error" in result and self.save_errors:
                    self.log_error(entry, result["error"])
                    with self.stats_lock:
                        self.stats["errors"] += 1
                
            except Exception as e:
                logger.error(f"Error processing segment: {e}")
                entry[self.corrected_text_field] = original_text
                entry[self.confidence_field] = 0.0
                entry[self.correction_made_field] = False
                with self.stats_lock:
                    self.stats["errors"] += 1
                
                if self.save_errors:
                    self.log_error(entry, str(e))
        
        return entry
    
    def log_error(self, entry: Dict[str, Any], error: str):
        """Log errors to file for analysis."""
        error_entry = {
            "segment_id": entry.get("segment_id", "unknown"),
            "audio_filepath": entry.get(self.audio_filepath_field),
            "original_text": entry.get(self.original_text_field),
            "error": error,
            "timestamp": time.time()
        }
        
        try:
            # Append to error log
            with open(self.error_log_file, "a") as f:
                f.write(json.dumps(error_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write error log: {e}")
    
    def process_entry_with_semaphore(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process entry with rate limiting semaphore.
        """
        with self.rate_limiter:
            return self.process_dataset_entry(entry)
    
    def process(self):
        """
        Process all entries in the manifest with parallel processing.
        """
        logger.info(f"Starting Gemini audio correction processing...")
        logger.info(f"Input: {self.input_manifest_file}")
        logger.info(f"Output: {self.output_manifest_file}")
        logger.info(f"Using {self.max_workers} parallel workers")
        logger.info(f"Rate limit: {self.requests_per_minute} requests per minute")
        
        # Load all entries
        entries = []
        with open(self.input_manifest_file, 'r') as f:
            for line in f:
                entries.append(json.loads(line))
        
        total_entries = len(entries)
        logger.info(f"Processing {total_entries} entries...")
        
        # Process entries in parallel
        processed_entries = [None] * total_entries  # Maintain order
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.process_entry_with_semaphore, entry): i 
                for i, entry in enumerate(entries)
            }
            
            # Process completed tasks
            completed = 0
            start_time = time.time()
            
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    processed_entries[index] = result
                    completed += 1
                    
                    # Log progress
                    if completed % 10 == 0 or completed == total_entries:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total_entries - completed) / rate if rate > 0 else 0
                        logger.info(
                            f"Progress: {completed}/{total_entries} "
                            f"({completed/total_entries*100:.1f}%) "
                            f"Rate: {rate:.1f}/s "
                            f"ETA: {eta:.0f}s"
                        )
                except Exception as e:
                    logger.error(f"Error processing entry {index}: {e}")
                    # Keep original entry on error
                    processed_entries[index] = entries[index]
        
        # Write output in original order
        with open(self.output_manifest_file, 'w') as f:
            for entry in processed_entries:
                if entry:  # Skip None entries (shouldn't happen)
                    f.write(json.dumps(entry) + "\n")
        
        # Calculate final statistics
        elapsed_time = time.time() - start_time
        
        # Print statistics
        logger.info("="*60)
        logger.info("Gemini Audio Correction Statistics:")
        logger.info(f"  Total segments: {self.stats['total_processed']}")
        logger.info(f"  Corrections made: {self.stats['corrections_made']} "
                   f"({self.stats['corrections_made']/max(1, self.stats['total_processed'])*100:.1f}%)")
        logger.info(f"  Errors: {self.stats['errors']}")
        logger.info(f"  API calls: {self.stats['api_calls']}")
        logger.info(f"  Processing time: {elapsed_time:.1f}s")
        logger.info(f"  Average speed: {total_entries/elapsed_time:.1f} entries/second")
        logger.info("="*60)
        
        if self.save_errors and self.stats["errors"] > 0:
            logger.info(f"Error details saved to: {self.error_log_file}")