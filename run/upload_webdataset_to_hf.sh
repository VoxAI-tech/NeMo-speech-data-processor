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
pretty_name: El Jannah Drive-thru Conversations
size_categories:
- 1K<n<10K
---

# El Jannah Drive-thru Conversations Dataset

This dataset contains transcribed El Jannah drive-thru conversations from El Jannah Australia.

## Dataset Details

- **Language**: English (en)
- **Domain**: Drive-thru conversations
- **Format**: WebDataset (tar files)
- **Processing Pipeline**: ${PIPELINE_TYPE}
- **ASR Model**: VoxAI/whisper-large-v3-polish-ct2
- **Corrections**: $([ "$PIPELINE_TYPE" == "qwen" ] && echo "Qwen3-8B LLM" || echo "Gemini 2.5 Pro Audio")

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
