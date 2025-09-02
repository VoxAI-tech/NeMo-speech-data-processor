"""Menu-aware correction processor for drive-thru ASR transcriptions."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz, process
from tqdm import tqdm

from sdp.processors.base_processor import BaseProcessor


class MenuAwareCorrection(BaseProcessor):
    """
    Correct menu item misspellings in ASR transcriptions using fuzzy matching.
    
    This processor uses rapidfuzz to identify and correct menu item names
    in transcriptions based on a known menu vocabulary.
    """
    
    def __init__(
        self,
        output_manifest_file: str,
        input_manifest_file: str,
        menu_vocabulary_file: str,
        text_field: str = "text",
        corrected_field: str = "text_corrected",
        fuzzy_threshold: int = 80,
        context_window: int = 3,
        save_corrections: bool = True,
        **kwargs,
    ):
        """
        Initialize MenuAwareCorrection processor.
        
        Args:
            input_manifest_file: Input manifest path
            output_manifest_file: Output manifest path
            menu_vocabulary_file: Path to menu vocabulary JSON
            text_field: Field containing text to correct
            corrected_field: Field to store corrected text
            fuzzy_threshold: Minimum fuzzy match score (0-100)
            context_window: Words to consider for multi-word items
            save_corrections: Whether to save correction details
        """
        super().__init__(output_manifest_file, input_manifest_file, **kwargs)
        
        self.text_field = text_field
        self.corrected_field = corrected_field
        self.fuzzy_threshold = fuzzy_threshold
        self.context_window = context_window
        self.save_corrections = save_corrections
        
        # Load menu vocabulary
        with open(menu_vocabulary_file, 'r') as f:
            self.menu_vocab = json.load(f)
        
        # Create lookup structures
        self._build_lookup_structures()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_corrected': 0,
            'correction_types': {},
            'common_corrections': {}
        }
    
    def _build_lookup_structures(self):
        """Build efficient lookup structures for menu items."""
        # All menu items for fuzzy matching
        self.all_items = self.menu_vocab.get('all_items', [])
        
        # Direct corrections mapping
        self.direct_corrections = self.menu_vocab.get('corrections', {})
        
        # Multi-word items for context matching
        self.multi_word_items = [
            item for item in self.all_items 
            if len(item.split()) > 1
        ]
        
        # Single word items
        self.single_word_items = [
            item for item in self.all_items
            if len(item.split()) == 1
        ]
        
        # Common menu terms by category
        self.category_terms = {
            'sizes': self.menu_vocab.get('sizes', []),
            'modifiers': self.menu_vocab.get('modifiers', []),
            'chicken': ['chicken', 'breast', 'leg', 'quarter', 'half', 'whole'],
            'sauces': ['sauce', 'garlic', 'chilli', 'mayo', 'tahini', 'bbq']
        }
    
    def _correct_text(self, text: str) -> Tuple[str, List[Dict]]:
        """
        Correct menu items in text using fuzzy matching.
        CONSERVATIVE: Only correct obvious misspellings, preserve speaker's exact words.
        
        Args:
            text: Input text to correct
            
        Returns:
            Tuple of (corrected_text, list of corrections made)
        """
        if not text:
            return text, []
        
        corrections = []
        words = text.lower().split()
        corrected_words = []
        i = 0
        
        # List of articles and modifiers to preserve
        preserve_words = {'a', 'an', 'the', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'}
        
        while i < len(words):
            word = words[i]
            corrected = False
            
            # Skip punctuation-attached words for now
            word_clean = word.rstrip('.,!?;:')
            punctuation = word[len(word_clean):]
            
            # Only check direct corrections for obvious misspellings
            # Don't correct single/plural or articles
            if word_clean in self.direct_corrections and word_clean not in preserve_words:
                correction = self.direct_corrections[word_clean]
                # Don't correct if it's just adding/removing 's' for plural
                if not (abs(len(correction) - len(word_clean)) == 1 and 
                       (correction.endswith('s') or word_clean.endswith('s'))):
                    corrected_words.append(correction + punctuation)
                    corrections.append({
                        'original': word_clean,
                        'corrected': correction,
                        'type': 'direct',
                        'score': 100
                    })
                    corrected = True
                    i += 1
                    continue
            
            # Try multi-word matching for known misspellings only
            if not corrected and i < len(words) - 1:
                for window_size in range(min(self.context_window, len(words) - i), 0, -1):
                    phrase = ' '.join(words[i:i+window_size])
                    phrase_clean = phrase.rstrip('.,!?;:')
                    
                    # Check direct multi-word corrections
                    if phrase_clean in self.direct_corrections:
                        correction = self.direct_corrections[phrase_clean]
                        corrected_words.append(correction)
                        corrections.append({
                            'original': phrase_clean,
                            'corrected': correction,
                            'type': 'direct_phrase',
                            'score': 100
                        })
                        i += window_size
                        corrected = True
                        break
                    
                    # CONSERVATIVE: Only do fuzzy matching for known misspellings
                    # Check if this looks like a misspelled menu item (e.g., "tubuli", "falafal")
                    if self._is_likely_misspelling(phrase_clean):
                        if self.multi_word_items:
                            match = process.extractOne(
                                phrase_clean, 
                                self.multi_word_items,
                                scorer=fuzz.ratio
                            )
                            
                            # Use higher threshold for multi-word corrections
                            if match and match[1] >= 90:  # Higher threshold
                                corrected_words.append(match[0])
                                corrections.append({
                                    'original': phrase_clean,
                                    'corrected': match[0],
                                    'type': 'fuzzy_phrase',
                                    'score': match[1]
                                })
                                i += window_size
                                corrected = True
                                break
            
            # CONSERVATIVE: Only correct obvious misspellings for single words
            if not corrected:
                # Only correct if it's a likely misspelling
                if self._is_likely_misspelling(word_clean) and len(word_clean) > 3:
                    match = process.extractOne(
                        word_clean,
                        self.single_word_items,
                        scorer=fuzz.ratio
                    )
                    
                    # Use very high threshold for single word corrections
                    if match and match[1] >= 85:
                        # Additional validation for common words
                        if self._validate_correction(word_clean, match[0], match[1]):
                            corrected_words.append(match[0] + punctuation)
                            corrections.append({
                                'original': word_clean,
                                'corrected': match[0],
                                'type': 'fuzzy_word',
                                'score': match[1]
                            })
                            corrected = True
            
            if not corrected:
                corrected_words.append(word)
            
            i += 1
        
        # Reconstruct text preserving original case where possible
        corrected_text = self._preserve_case(text, ' '.join(corrected_words))
        
        return corrected_text, corrections
    
    def _is_likely_misspelling(self, word: str) -> bool:
        """
        Check if a word is likely a misspelling of a menu item.
        
        Args:
            word: Word to check
            
        Returns:
            Whether it's likely a misspelling
        """
        # Known misspelling patterns
        misspelling_patterns = [
            'tubul', 'tabul', 'tabool', 'tabboul',  # tabouli variations
            'falaf', 'fellaf', 'felaf',  # falafel variations
            'humm', 'homm', 'hum',  # hommous variations
            'colslaw', 'colesla',  # coleslaw variations
        ]
        
        # Check if word contains misspelling patterns
        word_lower = word.lower()
        for pattern in misspelling_patterns:
            if pattern in word_lower:
                return True
        
        # Check if it's in our direct corrections dictionary
        if word_lower in self.direct_corrections:
            return True
        
        return False
    
    def _validate_correction(self, original: str, corrected: str, score: int) -> bool:
        """
        Validate if a correction should be applied.
        CONSERVATIVE: Only correct obvious errors.
        
        Args:
            original: Original word
            corrected: Proposed correction
            score: Fuzzy match score
            
        Returns:
            Whether to apply the correction
        """
        # Don't correct if it's just plural/singular difference
        if abs(len(original) - len(corrected)) == 1:
            if original.endswith('s') and corrected == original[:-1]:
                return False
            if corrected.endswith('s') and original == corrected[:-1]:
                return False
        
        # Don't correct common English words
        common_words = {'the', 'and', 'with', 'for', 'can', 'want', 'need', 'have', 
                       'one', 'two', 'three', 'a', 'an', 'no', 'yes', 'is', 'it'}
        if original in common_words:
            return False
        
        # Always correct known misspellings
        if original in self.direct_corrections:
            return True
        
        # Only correct if high confidence
        return score >= 85
    
    def _preserve_case(self, original: str, corrected: str) -> str:
        """Preserve original case pattern in corrected text."""
        if original.isupper():
            return corrected.upper()
        elif original[0].isupper() if original else False:
            return corrected.capitalize()
        return corrected
    
    def process(self):
        """Main processing method."""
        from tqdm import tqdm
        import json
        
        with open(self.input_manifest_file, 'r') as f_in, \
             open(self.output_manifest_file, 'w') as f_out:
            
            lines = f_in.readlines()
            for line in tqdm(lines, desc="Processing menu corrections"):
                entry = json.loads(line)
                processed_entry = self.process_entry(entry)
                if processed_entry:
                    f_out.write(json.dumps(processed_entry) + '\n')
        
        # Report statistics
        self.finalize({})
    
    def process_entry(self, entry: Dict) -> Optional[Dict]:
        """Process a single manifest entry."""
        if self.text_field not in entry:
            return entry
        
        original_text = entry[self.text_field]
        corrected_text, corrections = self._correct_text(original_text)
        
        # Add corrected text
        entry[self.corrected_field] = corrected_text
        
        # Track statistics
        self.stats['total_processed'] += 1
        if corrections:
            self.stats['total_corrected'] += 1
            
            # Save correction details if requested
            if self.save_corrections:
                entry['menu_corrections'] = corrections
                entry['num_corrections'] = len(corrections)
            
            # Update statistics
            for correction in corrections:
                correction_type = correction['type']
                self.stats['correction_types'][correction_type] = \
                    self.stats['correction_types'].get(correction_type, 0) + 1
                
                # Track common corrections
                key = f"{correction['original']} -> {correction['corrected']}"
                self.stats['common_corrections'][key] = \
                    self.stats['common_corrections'].get(key, 0) + 1
        
        return entry
    
    def finalize(self, metrics: Dict) -> None:
        """Finalize processing and report statistics."""
        
        print("\n=== Menu Correction Statistics ===")
        print(f"Total entries processed: {self.stats['total_processed']}")
        print(f"Entries with corrections: {self.stats['total_corrected']}")
        
        if self.stats['total_processed'] > 0:
            correction_rate = (self.stats['total_corrected'] / self.stats['total_processed']) * 100
            print(f"Correction rate: {correction_rate:.2f}%")
        
        print("\nCorrection types:")
        for correction_type, count in self.stats['correction_types'].items():
            print(f"  {correction_type}: {count}")
        
        # Show top common corrections
        if self.stats['common_corrections']:
            print("\nTop 10 most common corrections:")
            sorted_corrections = sorted(
                self.stats['common_corrections'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            for correction, count in sorted_corrections:
                print(f"  {correction}: {count} times")