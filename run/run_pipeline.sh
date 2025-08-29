#!/bin/bash

# EJ AU WebDataset Pipeline Runner
# Processes drive-thru audio data through complete annotation pipeline

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
export CUDA_VISIBLE_DEVICES=0  # Adjust based on your GPU availability
SDP_DIR="$(pwd)"
CONFIG_PATH="${SDP_DIR}/dataset_configs/multilingual/granary"
CONFIG_NAME="ej_au_webdataset_pipeline.yaml"

# Input/Output paths
INPUT_MANIFEST="${SDP_DIR}/data/initial_manifest.json"
OUTPUT_DIR="${SDP_DIR}/outputs/ej_au_webdataset"
CACHE_DIR="${OUTPUT_DIR}/cache"

# Sample limit (can be overridden by command line argument)
# Usage: ./run/run_pipeline.sh 10    # Process only 10 files
# Usage: ./run/run_pipeline.sh       # Process all files (default)
MAX_SAMPLES=${1:--1}  # First argument or -1 for all files

echo -e "${GREEN}=== EJ AU WebDataset Pipeline ===${NC}"
echo "SDP Directory: ${SDP_DIR}"
echo "Config: ${CONFIG_PATH}/${CONFIG_NAME}"
echo "Input Manifest: ${INPUT_MANIFEST}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Cache Directory: ${CACHE_DIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES}"
if [ "$MAX_SAMPLES" = "-1" ]; then
    echo -e "${GREEN}Processing: ALL files${NC}"
else
    echo -e "${YELLOW}Processing: Limited to $MAX_SAMPLES files (testing mode)${NC}"
fi
echo ""

# Check if input manifest exists
if [ ! -f "${INPUT_MANIFEST}" ]; then
    echo -e "${RED}Error: Input manifest not found at ${INPUT_MANIFEST}${NC}"
    echo -e "${YELLOW}Please run the manifest generator first:${NC}"
    echo "  ./run/generate_manifest.sh"
    exit 1
fi

# Show manifest info
MANIFEST_LINES=$(wc -l < "${INPUT_MANIFEST}")
echo -e "${GREEN}Input manifest contains $MANIFEST_LINES audio files${NC}"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CACHE_DIR}"

# Set up Python path
export PYTHONPATH="${SDP_DIR}:${PYTHONPATH}"

# Check GPU availability
echo ""
echo -e "${GREEN}GPU Status:${NC}"
nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader
echo ""

# Run the pipeline
echo -e "${YELLOW}Starting pipeline execution...${NC}"
echo "This will process audio through all stages:"
echo "1. Audio conversion to 16kHz WAV"
echo "2. Language detection and filtering"
echo "3. Whisper transcription (2 passes)"
echo "4. Quality filtering and grooming"
echo "5. Punctuation restoration with Qwen3-8B"
echo "6. WebDataset creation"
echo ""

# Log file
LOG_FILE="${OUTPUT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

# Execute pipeline with logging
echo -e "${YELLOW}Running pipeline (output will be logged to ${LOG_FILE})${NC}"
uv run python "${SDP_DIR}/main.py" \
    --config-path "${CONFIG_PATH}" \
    --config-name "${CONFIG_NAME}" \
    input_manifest_file="${INPUT_MANIFEST}" \
    output_dir="${OUTPUT_DIR}" \
    sdp_dir="${SDP_DIR}" \
    cache_dir="${CACHE_DIR}" \
    params.max_samples="${MAX_SAMPLES}" \
    2>&1 | tee "${LOG_FILE}"

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=== Pipeline completed successfully ===${NC}"
    echo "Output directory: ${OUTPUT_DIR}"
    echo "Log file: ${LOG_FILE}"
    
    # Check for final manifest
    FINAL_MANIFEST="${OUTPUT_DIR}/en/manifest_final.json"
    if [ -f "${FINAL_MANIFEST}" ]; then
        SEGMENT_COUNT=$(wc -l < "${FINAL_MANIFEST}")
        echo -e "${GREEN}✓ Final manifest: ${FINAL_MANIFEST} (${SEGMENT_COUNT} segments)${NC}"
    fi
    
    # Check for WebDataset files
    WEBDATASET_DIR="${OUTPUT_DIR}/en/webdataset"
    if [ -d "${WEBDATASET_DIR}" ]; then
        TAR_COUNT=$(find "${WEBDATASET_DIR}" -name "*.tar" | wc -l)
        TOTAL_SIZE=$(du -sh "${WEBDATASET_DIR}" | cut -f1)
        echo -e "${GREEN}✓ WebDataset: ${TAR_COUNT} tar files (${TOTAL_SIZE}) in ${WEBDATASET_DIR}${NC}"
    fi
else
    echo ""
    echo -e "${RED}=== Pipeline failed ===${NC}"
    echo -e "${YELLOW}Check log file for errors: ${LOG_FILE}${NC}"
    
    # Show last few error lines
    echo ""
    echo -e "${YELLOW}Last error lines:${NC}"
    tail -20 "${LOG_FILE}" | grep -E "(Error|ERROR|Failed|FAILED)" || true
    exit 1
fi