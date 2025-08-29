#!/bin/bash

# Upload EJ AU Structured Dataset to HuggingFace
# Usage: ./upload_to_hf.sh [dataset_name] [hf_username]

set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

DATASET_NAME=${1:-"ej-au-drivethru"}
HF_USERNAME=${2:-"your-username"}
DATASET_DIR="outputs/ej_au_webdataset/en/hf_dataset"

echo -e "${GREEN}=== HuggingFace Dataset Upload ===${NC}"
echo "Dataset Name: $DATASET_NAME"
echo "HuggingFace Username: $HF_USERNAME"
echo "Dataset Directory: $DATASET_DIR"
echo ""

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo -e "${RED}Error: Dataset directory not found: $DATASET_DIR${NC}"
    echo "Please run the pipeline first: bash run/run_pipeline.sh"
    exit 1
fi

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${YELLOW}Installing huggingface-cli...${NC}"
    pip install huggingface-hub
fi

# Check if user is logged in to HuggingFace
echo -e "${YELLOW}Checking HuggingFace authentication...${NC}"
if ! huggingface-cli whoami &> /dev/null; then
    echo -e "${YELLOW}Please login to HuggingFace:${NC}"
    huggingface-cli login
fi

# Count files
SPEECH_COUNT=$(find $DATASET_DIR/file/speech -name "*.wav" 2>/dev/null | wc -l || echo 0)
DIALOGUE_COUNT=$(find $DATASET_DIR/file/dialogue -name "*.json" 2>/dev/null | wc -l || echo 0)

echo -e "${GREEN}Dataset Statistics:${NC}"
echo "- Speech segments (WAV files): $SPEECH_COUNT"
echo "- Dialogue entries (JSON files): $DIALOGUE_COUNT"
echo ""

# Create dataset card
echo -e "${YELLOW}Creating dataset card...${NC}"
cat > $DATASET_DIR/README.md << EOF
---
license: apache-2.0
task_categories:
  - automatic-speech-recognition
  - text-to-speech
language:
  - en
tags:
  - drive-thru
  - conversational
  - speech
  - audio
  - whisper
pretty_name: EJ AU Drive-Thru Dataset
size_categories:
  - n<1K
---

# EJ AU Drive-Thru Dataset

## Dataset Description

This dataset contains drive-thru audio conversations with high-quality transcriptions.

### Dataset Statistics
- Speech segments: $SPEECH_COUNT
- Dialogue sessions: $DIALOGUE_COUNT
- Language: English
- Audio format: 16kHz mono WAV

## Dataset Structure

\`\`\`
file/
├── speech/           # Individual speech segments
│   └── {device_id}/
│       └── {session_id}/
│           ├── segment_XXXX.wav   # Audio file
│           └── segment_XXXX.json  # Metadata with transcription
└── dialogue/         # Complete dialogue sessions
    └── {device_id}/
        └── {session_id}/
            └── {session_id}.json  # All segments metadata
\`\`\`

## Audio Processing Pipeline

1. **Audio Conversion**: 16kHz mono WAV format
2. **Language Detection**: English filter (>0.7 confidence)
3. **Transcription**: Whisper Large-v3
4. **Punctuation Restoration**: Qwen3-8B
5. **Quality Filtering**: Hallucination detection and validation

## Usage

### Loading with Datasets Library

\`\`\`python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("$HF_USERNAME/$DATASET_NAME")

# Access speech segments
for sample in dataset['train']:
    audio = sample['audio']
    text = sample['text']
    metadata = sample['metadata']
\`\`\`

### Direct File Access

\`\`\`python
import json
import soundfile as sf

# Load a speech segment
segment_path = "file/speech/{device_id}/{session_id}/segment_0000"
audio, sr = sf.read(f"{segment_path}.wav")
with open(f"{segment_path}.json", 'r') as f:
    metadata = json.load(f)
    transcription = metadata['text']
\`\`\`

## Data Fields

Each speech segment JSON contains:
- \`text\`: Transcription with punctuation
- \`duration\`: Segment duration in seconds
- \`offset\`: Start time in original audio
- \`segment_id\`: Unique segment identifier
- \`source_lang\`: Language code (en)
- \`speaker\`: Speaker role (customer/employee/unknown)
- \`sid\`: Session ID
- \`device_id\`: Recording device identifier
- \`is_speech_segment\`: Boolean flag for speech
- \`emotion\`: Emotion tag (if available)
- \`pnc\`: Punctuation flag
- \`itn\`: ITN (Inverse Text Normalization) flag

## Citation

If you use this dataset, please cite:

\`\`\`bibtex
@dataset{ej_au_drivethru_2024,
  title={EJ AU Drive-Thru Audio Dataset},
  year={2024},
  author={Your Organization},
  publisher={HuggingFace},
}
\`\`\`

## License

Apache 2.0

## Acknowledgments

Processed using NeMo Speech Data Processor with Whisper and Qwen models.
EOF

echo -e "${GREEN}Dataset card created!${NC}"
echo ""

# Upload to HuggingFace
echo -e "${YELLOW}Uploading dataset to HuggingFace...${NC}"
echo -e "${YELLOW}This may take a while depending on dataset size...${NC}"

# Create the repository if it doesn't exist (force private for VoxAI)
echo -e "${YELLOW}Creating private repository...${NC}"
huggingface-cli repo create $DATASET_NAME --repo-type dataset --private 2>/dev/null || true

# Upload the dataset
cd $DATASET_DIR
echo -e "${YELLOW}Uploading files to private dataset...${NC}"
huggingface-cli upload $HF_USERNAME/$DATASET_NAME . --repo-type dataset --private

echo ""
echo -e "${GREEN}=== Upload Complete! ===${NC}"
echo -e "${GREEN}Dataset URL: https://huggingface.co/datasets/$HF_USERNAME/$DATASET_NAME${NC}"
echo ""
echo "Next steps:"
echo "1. Visit the dataset page to verify upload"
echo "2. Update dataset visibility settings if needed"
echo "3. Add additional documentation or examples"