#!/bin/bash

# Upload WebDataset to Hugging Face Hub
# Usage: ./upload_webdataset_to_hf.sh <dataset_path> [hf_repo_id]
# Example: ./upload_webdataset_to_hf.sh outputs/vox_pipeline_pl_qwen_full/pl/webdataset username/dataset-name

set -e

# Check if dataset path is provided
if [ $# -lt 1 ]; then
    echo "Error: Dataset path is required"
    echo "Usage: $0 <dataset_path> [hf_repo_id]"
    echo "Example: $0 outputs/vox_pipeline_pl_qwen_full/pl/webdataset username/bk-pl-drive-thru"
    exit 1
fi

DATASET_PATH="$1"
HF_REPO_ID="${2:-}"  # Optional, will prompt if not provided

# Verify dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset path does not exist: $DATASET_PATH"
    exit 1
fi

# Check for .tar files in the dataset path
TAR_COUNT=$(find "$DATASET_PATH" -name "*.tar" 2>/dev/null | wc -l)
if [ "$TAR_COUNT" -eq 0 ]; then
    echo "Error: No .tar files found in $DATASET_PATH"
    echo "Make sure the WebDataset conversion was successful"
    exit 1
fi

echo "========================================"
echo "WebDataset Upload to Hugging Face Hub"
echo "========================================"
echo "Dataset path: $DATASET_PATH"
echo "Found $TAR_COUNT tar files to upload"
echo ""

# Get dataset size
DATASET_SIZE=$(du -sh "$DATASET_PATH" | cut -f1)
echo "Total dataset size: $DATASET_SIZE"
echo ""

# If HF repo ID not provided, prompt for it
if [ -z "$HF_REPO_ID" ]; then
    echo "Enter Hugging Face repository ID (format: username/dataset-name):"
    read -r HF_REPO_ID
    
    if [ -z "$HF_REPO_ID" ]; then
        echo "Error: Repository ID is required"
        exit 1
    fi
fi

echo "Target repository: $HF_REPO_ID"
echo ""

# Check if user is logged in to Hugging Face
echo "Checking Hugging Face authentication..."
if ! uv run hf auth whoami &> /dev/null; then
    echo "Not logged in to Hugging Face. Please log in:"
    uv run hf auth login
fi

# Get logged-in username
HF_USERNAME=$(uv run hf auth whoami 2>/dev/null | grep -E "username:|Username:" | awk '{print $2}' || echo "unknown")
echo "Logged in as: $HF_USERNAME"
echo ""

# Extract dataset name from path for card creation
DATASET_NAME=$(basename "$(dirname "$DATASET_PATH")")
PIPELINE_TYPE=$(echo "$DATASET_PATH" | grep -o "qwen\|gemini" || echo "unknown")

# Create a README for the dataset
README_PATH="$DATASET_PATH/README.md"
cat > "$README_PATH" << EOMD
---
language:
- pl
license: apache-2.0
task_categories:
- automatic-speech-recognition
- text-generation
pretty_name: Polish Drive-thru Conversations
size_categories:
- 1K<n<10K
---

# Polish Drive-thru Conversations Dataset

This dataset contains transcribed Polish drive-thru conversations from Burger King Poland.

## Dataset Details

- **Language**: Polish (pl)
- **Domain**: Drive-thru conversations
- **Format**: WebDataset (tar files)
- **Processing Pipeline**: ${PIPELINE_TYPE}
- **ASR Model**: VoxAI/whisper-large-v3-polish-ct2
- **Corrections**: $([ "$PIPELINE_TYPE" == "qwen" ] && echo "Qwen3-8B LLM" || echo "Gemini 2.5 Pro Audio")

## Dataset Structure

Each sample contains:
- \`audio\`: Audio file (WAV format, 16kHz, mono)
- \`text\`: Transcribed and corrected text
- \`metadata\`: Additional information including:
  - \`duration\`: Audio duration in seconds
  - \`segment_id\`: Unique segment identifier
  - \`session_id\`: Conversation session ID
  - \`device_id\`: Recording device ID
  - \`audio_type\`: Speaker type (customer/employee)
  - \`dataset\`: Source dataset identifier
  - \`language\`: Language code

## Processing Pipeline

1. Audio conversion to 16kHz mono
2. Language detection and filtering (Polish only)
3. ASR transcription using Polish-optimized Whisper model
4. Hallucination detection and filtering
5. Menu-aware correction with fuzzy matching
6. LLM-based corrections for:
   - Menu item spelling
   - Polish language specific errors
   - Punctuation restoration
7. Text normalization

## Usage

\`\`\`python
from webdataset import WebDataset

# Load the dataset
dataset = WebDataset("path/to/dataset/shard-{000000..999999}.tar")

for sample in dataset:
    audio = sample["audio"]
    text = sample["text"]
    metadata = sample["json"]
\`\`\`

## License

This dataset is released under the Apache 2.0 License.
EOMD

echo "Created dataset card: $README_PATH"
echo ""

# Ask for confirmation
echo "========================================"
echo "Ready to upload to Hugging Face Hub"
echo "========================================"
echo "Repository: $HF_REPO_ID"
echo "Files to upload:"
find "$DATASET_PATH" -name "*.tar" -o -name "*.json" -o -name "*.md" | head -10
if [ "$TAR_COUNT" -gt 10 ]; then
    echo "... and $(($TAR_COUNT - 10)) more tar files"
fi
echo ""
echo -n "Do you want to proceed? (y/N): "
read -r CONFIRM

if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Upload cancelled"
    exit 0
fi

# Create the repository if it doesn't exist
echo ""
echo "Creating/verifying repository..."
uv run hf repo create "$HF_REPO_ID" --repo-type dataset --private --exist-ok

# Upload the dataset
echo ""
echo "Starting upload..."
echo "This may take a while depending on dataset size and internet speed..."

# Use new hf upload command for better handling of large files
uv run hf upload \
    "$HF_REPO_ID" \
    "$DATASET_PATH" \
    . \
    --repo-type dataset \
    --commit-message "Upload Polish drive-thru WebDataset"

echo ""
echo "========================================"
echo "Upload completed successfully!"
echo "========================================"
echo "Dataset available at: https://huggingface.co/datasets/$HF_REPO_ID"
echo ""
echo "To use the dataset:"
echo "  from datasets import load_dataset"
echo "  dataset = load_dataset('$HF_REPO_ID', streaming=True)"
echo "========================================"
