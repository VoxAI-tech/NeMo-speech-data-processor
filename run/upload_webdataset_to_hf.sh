#!/bin/bash

# Upload WebDataset to HuggingFace
# Usage: ./upload_webdataset_to_hf.sh [dataset_name] [hf_username]

set -e

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values - adjust as needed
DATASET_NAME=${1:-"ej-webdataset"}
HF_USERNAME=${2:-"VoxAI"}

# Find the most recent WebDataset directory
LATEST_OUTPUT=$(ls -td outputs/vox_pipeline_*/en/webdataset 2>/dev/null | head -1)

if [ -z "$LATEST_OUTPUT" ]; then
    echo -e "${RED}Error: No WebDataset found in outputs/vox_pipeline_*/en/webdataset${NC}"
    echo "Please run the pipeline first: bash run/run_vox_pipeline.sh"
    exit 1
fi

DATASET_DIR="$LATEST_OUTPUT"

echo -e "${GREEN}=== HuggingFace WebDataset Upload ===${NC}"
echo "Dataset Name: $DATASET_NAME"
echo "HuggingFace Username: $HF_USERNAME"
echo "Dataset Directory: $DATASET_DIR"
echo ""

# Check if dataset directory exists and has files
if [ ! -d "$DATASET_DIR" ]; then
    echo -e "${RED}Error: Dataset directory not found: $DATASET_DIR${NC}"
    exit 1
fi

# Count TAR files
TAR_COUNT=$(find "$DATASET_DIR/train" -name "*.tar" 2>/dev/null | wc -l || echo 0)

if [ "$TAR_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No TAR files found in $DATASET_DIR/train${NC}"
    exit 1
fi

echo -e "${GREEN}WebDataset Statistics:${NC}"
echo "- TAR shards: $TAR_COUNT"

# Calculate total size
TOTAL_SIZE=$(du -sb "$DATASET_DIR/train" | awk '{print $1}')
TOTAL_SIZE_MB=$((TOTAL_SIZE / 1024 / 1024))
echo "- Total size: ${TOTAL_SIZE_MB} MB"

# Check if dataset_metadata.json exists and show stats
METADATA_FILE="$DATASET_DIR/dataset_metadata.json"
if [ -f "$METADATA_FILE" ]; then
    echo ""
    echo -e "${GREEN}Dataset Metadata:${NC}"
    # Extract key stats using Python
    python3 -c "
import json
with open('$METADATA_FILE', 'r') as f:
    metadata = json.load(f)
    print(f\"- Total shards: {metadata.get('total_shards', 'N/A')}\")
    print(f\"- Total samples: {metadata.get('total_samples', 'N/A')}\")
    print(f\"- Total duration: {metadata.get('total_duration_hours', 0):.2f} hours\")
    print(f\"- Audio types: {metadata.get('audio_types', {})}\")
    print(f\"- Shuffled: {metadata.get('shuffled', False)}\")
    print(f\"- Slice with offset: {metadata.get('slice_with_offset', False)}\")
"
fi

echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed${NC}"
    echo "Please install uv first: https://github.com/astral-sh/uv"
    exit 1
fi

# Install huggingface-hub if needed
echo -e "${YELLOW}Checking dependencies...${NC}"
uv pip install huggingface-hub click --quiet 2>/dev/null || true

# Check if user is logged in to HuggingFace
echo -e "${YELLOW}Checking HuggingFace authentication...${NC}"
# Try new hf command first, fall back to huggingface-cli if needed
if command -v hf &> /dev/null; then
    if ! hf auth whoami &> /dev/null; then
        echo -e "${YELLOW}Please login to HuggingFace:${NC}"
        hf auth login
    fi
elif command -v huggingface-cli &> /dev/null; then
    if ! huggingface-cli whoami &> /dev/null; then
        echo -e "${YELLOW}Please login to HuggingFace:${NC}"
        huggingface-cli login
    fi
else
    echo -e "${YELLOW}Warning: HuggingFace CLI not found. Assuming logged in.${NC}"
fi

# Construct the full repository ID
FULL_REPO_ID="$HF_USERNAME/$DATASET_NAME"

echo ""
echo -e "${YELLOW}Ready to upload WebDataset to: $FULL_REPO_ID${NC}"
echo -e "${YELLOW}The repository will be created as private by default.${NC}"
echo ""

# Ask for confirmation
read -p "Do you want to proceed with the upload? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled."
    exit 0
fi

# Run the upload script
echo ""
echo -e "${YELLOW}Uploading WebDataset to HuggingFace...${NC}"
echo -e "${YELLOW}This may take a while depending on dataset size...${NC}"

uv run python scripts/push_webdataset_to_hf.py "$DATASET_DIR" "$FULL_REPO_ID"

echo ""
echo -e "${GREEN}=== Upload Complete! ===${NC}"
echo -e "${GREEN}Dataset URL: https://huggingface.co/datasets/$FULL_REPO_ID${NC}"
echo ""
echo "Next steps:"
echo "1. Visit the dataset page to verify upload"
echo "2. Update dataset visibility settings if needed (currently private)"
echo "3. Test loading the dataset with WebDataset:"
echo "   import webdataset as wds"
echo "   dataset = wds.WebDataset('https://huggingface.co/datasets/$FULL_REPO_ID/resolve/main/train/*.tar')"
echo "4. Or with HuggingFace datasets:"
echo "   from datasets import load_dataset"
echo "   dataset = load_dataset('$FULL_REPO_ID', streaming=True)"