#!/bin/bash

# Script to upload EJ AU WebDataset to HuggingFace Hub
# Usage: ./upload_ej_au_to_hf.sh <hf_username_or_org> [token]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <hf_username_or_org> [hf_token]"
    echo "Example: $0 myusername"
    echo "Example with token: $0 myusername hf_xxxxxxxxxxxxx"
    exit 1
fi

HF_USER_OR_ORG=$1
HF_TOKEN=${2:-}  # Optional token, use environment variable or login if not provided
DATASET_NAME="ej-au-drive-thru-asr"
REPO_ID="${HF_USER_OR_ORG}/${DATASET_NAME}"

# Path to WebDataset tar files
TAR_DIR="outputs/ej_au_webdataset/en/webdataset"

# Dataset metadata
SUBSET="drive-thru"  # Subset name for organization
SPLIT="train"        # Using train split

echo "==========================================="
echo "Uploading EJ AU Drive-thru Dataset to HuggingFace"
echo "==========================================="
echo "Repository: ${REPO_ID}"
echo "Tar directory: ${TAR_DIR}"
echo "Subset: ${SUBSET}"
echo "Split: ${SPLIT}"
echo ""

# Check if tar directory exists
if [ ! -d "$TAR_DIR" ]; then
    echo "Error: WebDataset directory not found at ${TAR_DIR}"
    echo "Please run the pipeline first to generate WebDataset files."
    exit 1
fi

# Count tar files
TAR_COUNT=$(ls ${TAR_DIR}/*.tar 2>/dev/null | wc -l)
if [ "$TAR_COUNT" -eq 0 ]; then
    echo "Error: No tar files found in ${TAR_DIR}"
    exit 1
fi

echo "Found ${TAR_COUNT} tar files to upload"
echo ""

# Dry run first to verify
echo "Running dry run to verify upload structure..."
if [ -n "$HF_TOKEN" ]; then
    uv run scripts/push_webdataset_to_hf.py \
        "${TAR_DIR}" \
        "${REPO_ID}" \
        --token "${HF_TOKEN}" \
        --subset "${SUBSET}" \
        --split "${SPLIT}" \
        --dry-run
else
    uv run scripts/push_webdataset_to_hf.py \
        "${TAR_DIR}" \
        "${REPO_ID}" \
        --subset "${SUBSET}" \
        --split "${SPLIT}" \
        --dry-run
fi

echo ""
read -p "Proceed with upload? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting upload..."
    if [ -n "$HF_TOKEN" ]; then
        uv run scripts/push_webdataset_to_hf.py \
            "${TAR_DIR}" \
            "${REPO_ID}" \
            --token "${HF_TOKEN}" \
            --subset "${SUBSET}" \
            --split "${SPLIT}"
    else
        uv run scripts/push_webdataset_to_hf.py \
            "${TAR_DIR}" \
            "${REPO_ID}" \
            --subset "${SUBSET}" \
            --split "${SPLIT}"
    fi
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Upload complete!"
        echo "Dataset available at: https://huggingface.co/datasets/${REPO_ID}"
        echo ""
        echo "To load the dataset in Python:"
        echo "  from datasets import load_dataset"
        echo "  dataset = load_dataset('${REPO_ID}', name='${SUBSET}', split='${SPLIT}')"
    else
        echo "Upload failed. Check error messages above."
    fi
else
    echo "Upload cancelled."
fi