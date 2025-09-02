"""Cross-channel validation processor for dual-channel drive-thru audio."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from rapidfuzz import fuzz
from tqdm import tqdm

from sdp.processors.base_processor import BaseProcessor


class CrossChannelValidation(BaseProcessor):
    """
    Validate and merge transcriptions from dual audio channels (mic/spk).
    
    This processor compares transcriptions from customer (mic) and employee (spk)
    channels to improve accuracy through cross-validation.
    """
    
    def __init__(
        self,
        output_manifest_file: str,
        input_manifest_file: str,
        paired_manifest_file: Optional[str] = None,
        text_field: str = "text",
        validated_field: str = "text_validated",
        confidence_field: str = "cross_channel_confidence",
        similarity_threshold: float = 0.7,
        use_cleaner_channel: bool = True,
        merge_strategy: str = "best_confidence",
        **kwargs,
    ):
        """
        Initialize CrossChannelValidation processor.
        
        Args:
            input_manifest_file: Input manifest path
            output_manifest_file: Output manifest path
            paired_manifest_file: Optional manifest with paired channel data
            text_field: Field containing transcription text
            validated_field: Field to store validated text
            confidence_field: Field to store cross-channel confidence
            similarity_threshold: Minimum similarity for validation (0-1)
            use_cleaner_channel: Prefer cleaner (spk) channel when similar
            merge_strategy: How to merge channels ('best_confidence', 'speaker_priority', 'consensus')
        """
        super().__init__(output_manifest_file, input_manifest_file, **kwargs)
        
        self.text_field = text_field
        self.validated_field = validated_field
        self.confidence_field = confidence_field
        self.similarity_threshold = similarity_threshold * 100  # Convert to rapidfuzz scale
        self.use_cleaner_channel = use_cleaner_channel
        self.merge_strategy = merge_strategy
        
        # Load paired manifest if provided
        self.paired_data = {}
        if paired_manifest_file:
            self._load_paired_manifest(paired_manifest_file)
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'validated': 0,
            'cross_validated': 0,
            'low_confidence': 0,
            'channel_agreement': defaultdict(int),
            'merge_decisions': defaultdict(int)
        }
    
    def _load_paired_manifest(self, manifest_file: str):
        """Load paired channel data from manifest."""
        with open(manifest_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # Create lookup key based on session and timing
                key = self._create_pair_key(entry)
                if key:
                    self.paired_data[key] = entry
    
    def _create_pair_key(self, entry: Dict) -> Optional[str]:
        """Create a key for matching paired channel entries."""
        if all(k in entry for k in ['session_id', 'offset', 'duration']):
            # Match entries from same session with similar timing
            return f"{entry['session_id']}_{entry['offset']:.2f}_{entry['duration']:.2f}"
        return None
    
    def _find_paired_entry(self, entry: Dict) -> Optional[Dict]:
        """Find the paired channel entry for cross-validation."""
        # First try exact match
        key = self._create_pair_key(entry)
        if key and key in self.paired_data:
            return self.paired_data[key]
        
        # Try fuzzy time matching for same session
        if 'session_id' in entry:
            session_id = entry['session_id']
            offset = entry.get('offset', 0)
            duration = entry.get('duration', 0)
            
            # Look for overlapping segments in same session
            for pair_key, pair_entry in self.paired_data.items():
                if session_id in pair_key:
                    pair_offset = pair_entry.get('offset', 0)
                    pair_duration = pair_entry.get('duration', 0)
                    
                    # Check for temporal overlap
                    overlap = self._calculate_overlap(
                        offset, offset + duration,
                        pair_offset, pair_offset + pair_duration
                    )
                    
                    if overlap > 0.5:  # At least 50% overlap
                        return pair_entry
        
        return None
    
    def _calculate_overlap(self, start1: float, end1: float, 
                          start2: float, end2: float) -> float:
        """Calculate temporal overlap ratio between two segments."""
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return 0.0
        
        overlap_duration = overlap_end - overlap_start
        min_duration = min(end1 - start1, end2 - start2)
        
        return overlap_duration / min_duration if min_duration > 0 else 0.0
    
    def _validate_transcriptions(self, text1: str, text2: str) -> Tuple[str, float]:
        """
        Validate and merge transcriptions from two channels.
        
        Args:
            text1: Transcription from first channel (mic)
            text2: Transcription from second channel (spk)
            
        Returns:
            Tuple of (validated_text, confidence_score)
        """
        if not text1 and not text2:
            return "", 0.0
        
        if not text1:
            return text2, 0.5
        
        if not text2:
            return text1, 0.5
        
        # Calculate similarity
        similarity = fuzz.ratio(text1.lower(), text2.lower())
        
        # High similarity - channels agree
        if similarity >= self.similarity_threshold:
            self.stats['channel_agreement']['high'] += 1
            
            if self.merge_strategy == 'speaker_priority' or self.use_cleaner_channel:
                # Prefer speaker channel (usually cleaner)
                self.stats['merge_decisions']['used_speaker'] += 1
                return text2, similarity / 100.0
            elif self.merge_strategy == 'consensus':
                # Merge by taking common words
                merged = self._consensus_merge(text1, text2)
                self.stats['merge_decisions']['consensus'] += 1
                return merged, similarity / 100.0
            else:  # best_confidence
                # Use the one that seems more complete
                if len(text2) > len(text1):
                    self.stats['merge_decisions']['used_speaker'] += 1
                    return text2, similarity / 100.0
                else:
                    self.stats['merge_decisions']['used_customer'] += 1
                    return text1, similarity / 100.0
        
        # Medium similarity - partial agreement
        elif similarity >= 50:
            self.stats['channel_agreement']['medium'] += 1
            
            # Try to extract common elements
            if self.merge_strategy == 'consensus':
                merged = self._consensus_merge(text1, text2)
                self.stats['merge_decisions']['partial_consensus'] += 1
                return merged, similarity / 100.0
            else:
                # Use cleaner channel with lower confidence
                if self.use_cleaner_channel:
                    self.stats['merge_decisions']['used_speaker_low_conf'] += 1
                    return text2, similarity / 100.0
                else:
                    self.stats['merge_decisions']['used_customer_low_conf'] += 1
                    return text1, similarity / 100.0
        
        # Low similarity - channels disagree
        else:
            self.stats['channel_agreement']['low'] += 1
            
            # Use the cleaner channel but mark low confidence
            if self.use_cleaner_channel:
                self.stats['merge_decisions']['fallback_speaker'] += 1
                return text2, similarity / 100.0
            else:
                self.stats['merge_decisions']['fallback_customer'] += 1
                return text1, similarity / 100.0
    
    def _consensus_merge(self, text1: str, text2: str) -> str:
        """Merge two texts by finding consensus words."""
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        # Use dynamic programming to find longest common subsequence
        merged_words = []
        i, j = 0, 0
        
        while i < len(words1) and j < len(words2):
            # Check if words match (exact or fuzzy)
            if words1[i] == words2[j] or fuzz.ratio(words1[i], words2[j]) > 85:
                merged_words.append(words1[i])
                i += 1
                j += 1
            else:
                # Look ahead for matches
                found_match = False
                
                # Check if word from text1 appears soon in text2
                for k in range(j + 1, min(j + 3, len(words2))):
                    if words1[i] == words2[k] or fuzz.ratio(words1[i], words2[k]) > 85:
                        # Add intermediate words from text2
                        merged_words.extend(words2[j:k])
                        j = k
                        found_match = True
                        break
                
                if not found_match:
                    # Check if word from text2 appears soon in text1
                    for k in range(i + 1, min(i + 3, len(words1))):
                        if words2[j] == words1[k] or fuzz.ratio(words2[j], words1[k]) > 85:
                            # Add intermediate words from text1
                            merged_words.extend(words1[i:k])
                            i = k
                            found_match = True
                            break
                
                if not found_match:
                    # No consensus, skip the word in text1
                    i += 1
        
        # Add remaining words if any
        if i < len(words1):
            merged_words.extend(words1[i:])
        elif j < len(words2):
            merged_words.extend(words2[j:])
        
        return ' '.join(merged_words)
    
    def process(self):
        """Main processing method."""
        from tqdm import tqdm
        import json
        
        with open(self.input_manifest_file, 'r') as f_in, \
             open(self.output_manifest_file, 'w') as f_out:
            
            lines = f_in.readlines()
            for line in tqdm(lines, desc="Processing cross-channel validation"):
                entry = json.loads(line)
                processed_entry = self.process_entry(entry)
                if processed_entry:
                    f_out.write(json.dumps(processed_entry) + '\n')
        
        # Report statistics
        self.finalize({})
    
    def process_entry(self, entry: Dict) -> Optional[Dict]:
        """Process a single manifest entry."""
        self.stats['total_processed'] += 1
        
        # Get current channel text
        current_text = entry.get(self.text_field, "")
        current_channel = entry.get('audio_type', 'unknown')
        
        # Find paired channel entry
        paired_entry = self._find_paired_entry(entry)
        
        if paired_entry:
            paired_text = paired_entry.get(self.text_field, "")
            
            # Validate and merge transcriptions
            validated_text, confidence = self._validate_transcriptions(
                current_text if current_channel == 'customer' else paired_text,
                paired_text if current_channel == 'customer' else current_text
            )
            
            entry[self.validated_field] = validated_text
            entry[self.confidence_field] = confidence
            entry['cross_validated'] = True
            
            self.stats['cross_validated'] += 1
            if confidence >= 0.7:
                self.stats['validated'] += 1
            else:
                self.stats['low_confidence'] += 1
            
            # Add validation metadata
            entry['validation_metadata'] = {
                'original_text': current_text,
                'paired_text': paired_text,
                'similarity': fuzz.ratio(current_text.lower(), paired_text.lower()),
                'merge_strategy': self.merge_strategy,
                'channel': current_channel
            }
        else:
            # No paired data, use original text
            entry[self.validated_field] = current_text
            entry[self.confidence_field] = 0.5  # Default confidence
            entry['cross_validated'] = False
            
            # Keep original text but mark as not validated
            entry['validation_metadata'] = {
                'original_text': current_text,
                'paired_text': None,
                'similarity': 0,
                'merge_strategy': 'no_pair',
                'channel': current_channel
            }
        
        return entry
    
    def finalize(self, metrics: Dict) -> None:
        """Finalize processing and report statistics."""
        
        print("\n=== Cross-Channel Validation Statistics ===")
        print(f"Total entries processed: {self.stats['total_processed']}")
        print(f"Cross-validated entries: {self.stats['cross_validated']}")
        print(f"High confidence validations: {self.stats['validated']}")
        print(f"Low confidence entries: {self.stats['low_confidence']}")
        
        if self.stats['total_processed'] > 0:
            validation_rate = (self.stats['cross_validated'] / self.stats['total_processed']) * 100
            print(f"Cross-validation rate: {validation_rate:.2f}%")
        
        print("\nChannel agreement distribution:")
        for level, count in self.stats['channel_agreement'].items():
            print(f"  {level}: {count}")
        
        print("\nMerge decisions:")
        for decision, count in self.stats['merge_decisions'].items():
            print(f"  {decision}: {count}")