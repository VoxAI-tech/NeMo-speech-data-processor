# Vox Pipeline Documentation

## Overview
The Vox Pipeline is a comprehensive ASR training data processing system specifically designed for drive-thru conversation audio. It implements a 32-stage pipeline that transforms raw multi-channel audio recordings into high-quality training datasets for Whisper and other ASR models.

## Key Capabilities

- **Dual-channel processing:** Handles customer (mic) and employee (spk) audio separately
- **Polish language optimization:** Specialized for Polish ASR with voxai/whisper-large-v3-polish-ct2
- **Menu-aware correction:** Two-stage correction using fuzzy matching and audio verification
- **Quality filtering:** Multi-stage filtering for hallucinations, language, and duration
- **WebDataset output:** Efficient tar archive format for training

## Data Structure

### Polish Drive-Thru (Burger King Poland)
```
data/audio/
└── {device_id}/
    └── {session_id}/
        ├── mic.wav  (customer audio - noisy)
        └── spk.wav  (employee audio - clean)

Devices: 0304UD41, 1200UD26, 1840UD05, 1853UD05
```

## Complete 32-Stage Pipeline Architecture

### Phase 1: Audio Preparation (Stages 0-4)

#### Stage 0: CreateInitialManifestVox
Creates initial manifest from Vox directory structure.
- Scans directory for audio files
- Extracts metadata (device_id, session_id, channel type)
- Creates initial JSON manifest

#### Stage 1: Audio ID Generation
Adds unique identifier for tracking through pipeline.
- Format: `{device_id}/{session_id}/{channel}`

#### Stage 2: Sample Limiting (Optional)
Limits dataset size for testing.
- Default: -1 (all samples)
- Testing: Set specific number

#### Stage 3: Audio Conversion (FfmpegConvert)
Standardizes audio format.
- Target: 16kHz mono WAV
- Preserves directory structure
- Creates normalized audio copies

#### Stage 4: Duration Extraction
Calculates and adds audio duration metadata.

### Phase 2: Language Detection & Filtering (Stages 5-7)

#### Stage 5: Language Detection (FasterWhisperInference)
Uses Whisper large-v3 for language identification.
- Detection segments: 7
- Chunk length: 30 seconds
- Output: language probability scores

#### Stage 6: Language Filtering
Keeps only target language audio.
- Polish threshold: 70% confidence
- Filters out non-Polish segments

#### Stage 7: Field Cleanup
Removes temporary language detection fields.

### Phase 3: Initial Transcription (Stages 8-10)

#### Stage 8: Full Transcription (FasterWhisperInference)
First-pass transcription with segment boundaries.
- Model: Whisper large-v3 (or voxai/whisper-large-v3-polish-ct2)
- Creates segment timestamps
- Full conversation transcription

#### Stage 9: Duration Drop
Removes file-level duration (will be recalculated per segment).

#### Stage 9.5: Empty Segment Filter (Polish Pipeline)
Filters out entries with no segments.
- Prevents downstream errors
- Ensures all entries have transcribable content

#### Stage 10: Segment Expansion (ListToEntries)
**CRITICAL**: Converts segments to individual entries.
- 1 audio file with N segments → N manifest entries
- Each segment becomes separate training sample

### Phase 4: Segment Processing (Stages 11-15)

#### Stage 11: Field Selection
Keeps only necessary segment fields.

#### Stage 12: Duration Calculation
Calculates duration for each segment.
- Formula: `end_time - start_time`

#### Stage 13: Duration Filtering
Removes too short/long segments.
- Min: 0.5 seconds
- Max: 30 seconds (configurable)

#### Stage 14: Field Renaming
Standardizes field names.
- `start` → `offset`
- `id` → `segment_id`

#### Stage 15: Core Field Selection
Prepares for second-pass transcription.

### Phase 5: Refined Transcription (Stage 16-18)

#### Stage 16: Slice-by-Offset Transcription
Second-pass transcription on individual segments.
- More accurate than full-file transcription
- Uses offset/duration for precise extraction

#### Stage 17: Field Management
Keeps transcription results and metadata.

#### Stage 18: Text Field Renaming
- `pred_text` → `text`

### Phase 6: Quality Filtering (Stages 19-22)

#### Stage 19: Empty Text Removal
Drops segments with empty transcriptions.

#### Stage 20: Hallucination Detection
Identifies Whisper hallucinations.
- Repeated n-grams
- Long words
- Frequent single words

#### Stage 21: Hallucination Filtering
Removes detected hallucinations.

#### Stage 22: Field Cleanup
Removes hallucination detection fields.

### Phase 7: Text Correction (Stages 23-24)

#### Stage 23: Menu-Aware Fuzzy Correction
First-pass menu item correction.
- Fuzzy matching threshold: 80%
- Context window: 3 words
- Menu vocabulary: `assets/vocabularies/bk_menu_vocabulary.json`
- Preserves Polish words (kurczak, ser)

#### Stage 24: Gemini Audio Verification (Optional)
Audio-based spelling verification.
- Model: Gemini 2.5 Pro
- Listens to actual audio
- Verifies menu item pronunciations
- Rate limit: 145 RPM (Tier 1)
- Automatic caching for resume

### Phase 8: Text Enhancement (Stages 25-27)

#### Stage 25: Punctuation Restoration (vLLMInference)
Adds punctuation and capitalization.
- Model: Qwen3-8B
- Temperature: 0.3
- Preserves menu corrections

#### Stage 26: Clean LLM Output
Removes artifacts from Qwen generation.
- Strips thinking tokens
- Cleans formatting issues

#### Stage 27: Text Normalization (SubRegex)
Final text cleaning with regex patterns.
- Standardizes formatting
- Removes special characters

### Phase 9: Final Processing (Stages 28-32)

#### Stage 28: Drop Intermediate Fields
Removes processing artifacts.
- Intermediate text versions
- Correction metadata
- Temporary fields

#### Stage 29: Field Renaming
Final field standardization.
- `src_text` → `text`
- `offset` → `source_audio_offset`

#### Stage 30: Add Metadata
Adds constant fields.
- Dataset name: `bk_pl_drive_thru`
- Language: `pl`

#### Stage 31: Final Field Selection
Keeps only training-relevant fields:
- audio_filepath
- text
- duration
- source_audio_offset
- segment_id
- audio_type (customer/employee)
- device_id
- session_id
- dataset
- language

#### Stage 32: WebDataset Conversion
Creates tar archives for efficient training.
- Shard size: 100 files
- Format: WAV + JSON pairs
- Shuffled with seed 42
- Slice-with-offset enabled

## Configuration Files

### Main Configurations
- `config_pl_qwen.yaml` - Full Qwen pipeline (32 stages)
- `config_pl_gemini_from_qwen.yaml` - Gemini verification (stages 24-32)

### Supporting Files
- `assets/vocabularies/bk_menu_vocabulary.json` - Menu items and misspellings
- `assets/vocabularies/menu_vocabulary.json` - El Jannah menu vocabulary  
- `assets/menus/bk_menu.json` - Burger King menu data
- `assets/menus/el_jannah_menu.json` - El Jannah menu data
- `assets/scripts/extract_bk_menu_vocabulary.py` - BK vocabulary extractor
- `assets/scripts/extract_menu_vocabulary.py` - El Jannah vocabulary extractor
- `partials/gemini_audio_prompts/pl.yaml` - Gemini prompts
- `partials/pr_recovery_prompts/pl.yaml` - Punctuation prompts
- `partials/common_phrases/pl.txt` - Hallucination patterns

## Performance Metrics

### Processing Speed
- **Qwen Pipeline**: ~11,000 segments in 3-4 hours
- **Gemini Pipeline**: 145 requests/minute (limited by API)

### Quality Metrics
- **Language Detection**: 70% confidence threshold
- **Menu Correction**: 80% fuzzy match threshold
- **Duration Filter**: 0.5-30 seconds
- **Expected Output**: 70-80% retention rate

## Common Issues and Solutions

### Issue 1: Empty Segments Error
**Solution**: Added stage 9.5 to filter empty segment lists

### Issue 2: API Rate Limits
**Solution**: Implemented caching and automatic resume

### Issue 3: Polish Word Corrections
**Solution**: Preserved Polish words in menu vocabulary

### Issue 4: Long Processing Times
**Solution**: Created Gemini-from-Qwen pipeline to reuse stages 0-23

## Future Enhancements

Based on research in [proposal.md](proposal.md), potential improvements include:

1. **Cross-Channel Audio Processing**
   - Reduce audio bleed between mic/spk channels
   - Use both channels for validation

2. **Advanced VAD**
   - Better handling of 0.5-3 second utterances
   - Improved silence detection

3. **Multi-Pass Architecture**
   - Iterative refinement of transcriptions
   - Cross-validation between passes

4. **Whisper Contextual Biasing**
   - Domain adaptation without full fine-tuning
   - Menu vocabulary priming

5. **Active Learning**
   - Uncertainty-based sample selection
   - Quality-guided manual review

## Usage Examples

### Running Full Qwen Pipeline
```bash
./run/run_polish_pipeline_qwen_full.sh
```

### Running Gemini Verification
```bash
export GEMINI_API_KEY="your_key"
./run/run_polish_pipeline_gemini_from_qwen.sh
```

### Checking Progress
```bash
./run/check_gemini_progress.sh
```

### Uploading to HuggingFace
```bash
./run/upload_webdataset_to_hf.sh outputs/vox_pipeline_pl_qwen_full/pl/webdataset VoxAI/bk-pl-$(date +%Y%m%d)
```

## Output Format

Final manifest entry (manifest_32.json):
```json
{
  "audio_filepath": "outputs/.../converted_audio/1200UD26/.../mic.wav",
  "text": "Poproszę dużego whoppera z frytkami.",
  "duration": 2.5,
  "source_audio_offset": 43.12,
  "segment_id": 0,
  "audio_type": "customer",
  "device_id": "1200UD26",
  "session_id": "ed34d451-fe50-4d76-b644-8cc0f23b38b1",
  "dataset": "bk_pl_drive_thru",
  "language": "pl"
}
```

WebDataset structure:
```
webdataset/
└── train/
    ├── shard_00000.tar
    ├── shard_00001.tar
    └── ...
```

Each tar contains:
- `.wav` - Audio segment (extracted with offset/duration)
- `.json` - Metadata and transcription

## Menu Vocabulary Management

The pipeline includes automated menu vocabulary extraction to support different restaurant chains and languages. The assets folder contains organized menu data and extraction scripts.

### Assets Structure

```
dataset_configs/vox_pipeline/assets/
├── menus/                      # Raw menu data
│   ├── bk_menu.json           # Burger King menu items
│   └── el_jannah_menu.json    # El Jannah menu items
├── vocabularies/              # Generated vocabularies for correction
│   ├── bk_menu_vocabulary.json        # BK vocabulary with corrections
│   └── menu_vocabulary.json           # El Jannah vocabulary
└── scripts/                   # Vocabulary extraction tools
    ├── extract_bk_menu_vocabulary.py  # BK-specific extractor
    └── extract_menu_vocabulary.py     # General menu extractor
```

### Adding New Restaurant Menus

To add a new restaurant chain to the pipeline:

#### 1. Prepare Menu Data
Create a JSON file with the restaurant's menu structure in `assets/menus/`:

```json
[
  {
    "name": "Whopper Burger",
    "category": "burgers",
    "price": 12.50,
    "option_groups": [
      {
        "name": "size",
        "options": [
          {"name": "small", "price": 0},
          {"name": "large", "price": 2}
        ]
      }
    ]
  }
]
```

#### 2. Extract Vocabulary
Use the appropriate extraction script based on your menu format:

**For Burger King-style menus:**
```bash
cd dataset_configs/vox_pipeline/assets/scripts
python extract_bk_menu_vocabulary.py
```

**For El Jannah-style menus:**
```bash
cd dataset_configs/vox_pipeline/assets/scripts
python extract_menu_vocabulary.py
```

#### 3. Customize Vocabulary Extraction

The extraction scripts automatically:
- Categorize menu items (chicken, sauces, sides, drinks, etc.)
- Extract base items without size modifiers
- Include option variations
- Generate common misspelling corrections

**Key vocabulary categories:**
- `chicken_items` - Chicken dishes and parts
- `sauces` - All sauce variations
- `sides` - Side dishes and accompaniments
- `drinks` - Beverages
- `rolls_burgers` - Sandwich items
- `sizes` - Size modifiers (small, medium, large)
- `modifiers` - Request modifiers (add, no, extra, etc.)
- `corrections` - Misspelling → correct mappings

#### 4. Configure Pipeline

Update your pipeline configuration to use the new vocabulary:

```yaml
- _target_: sdp.processors.menu_aware_correction.MenuAwareCorrection
  menu_vocabulary_file: ${sdp_dir}/dataset_configs/vox_pipeline/assets/vocabularies/your_menu_vocabulary.json
```

### Best Practices for Menu Vocabularies

#### Conservative Corrections
Only include obvious misspellings in the corrections dictionary:

```json
{
  "corrections": {
    "whooper": "whopper",          // Clear misspelling
    "whoppera": "whopper",         // Clear misspelling
    "tubuli": "tabouli",          // Clear misspelling
    // DON'T include:
    // "garlic": "garlic sauce"    // Let users say abbreviated forms
    // "quarter": "1/4 chicken"   // Preserve natural speech
  }
}
```

#### Language Considerations
- **Polish**: Preserve Polish words (kurczak, ser) - don't auto-translate
- **English**: Keep natural abbreviations (garlic vs garlic sauce)
- **Arabic**: Include transliteration variations for Arabic menu items

#### Testing Vocabulary Changes
After updating vocabularies, test with small samples:

```bash
# Test vocabulary changes
python main.py \
  --config-path dataset_configs/vox_pipeline/ \
  --config-name config_pl_qwen.yaml \
  params.max_samples=10 \
  processors_to_run="22:24"  # Just run menu correction stages
```

### Supported Menu Formats

The extraction scripts handle various menu data structures:

1. **Flat item lists** (simple name/price pairs)
2. **Categorized menus** (items grouped by category)
3. **Option groups** (customizations and modifiers)
4. **Nested variations** (sizes, add-ons, combinations)

For custom menu formats, extend the extraction scripts or create new ones following the established patterns.