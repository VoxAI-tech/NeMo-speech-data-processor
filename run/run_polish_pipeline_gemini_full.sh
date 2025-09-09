#!/bin/bash

# Polish Drive-thru Pipeline with Gemini Audio Verification
# Full dataset processing version
# Uses Gemini 2.5 Pro for audio-based corrections

set -e

# Configuration
DATA_DIR="data/audio"
OUTPUT_DIR="outputs/vox_pipeline_pl_gemini_full"
SDP_DIR="$(pwd)"
# Process ALL samples - no limit
MAX_SAMPLES=10  # REMOVE THIS LINE TO PROCESS FULL DATASET
TOKENIZERS_PARALLELISM=false
echo "========================================"
echo "Polish Pipeline - Gemini Audio Version (FULL DATASET)"
echo "========================================"
echo "Model: VoxAI/whisper-large-v3-polish-ct2"
echo "Correction: Gemini 2.5 Pro (audio-based)"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Samples: ALL (no limit)"
echo "========================================"
echo ""
echo "WARNING: This will process the ENTIRE dataset (7,391 files)"
echo "Estimated time: 8-12 hours (slower than Qwen due to audio verification)"
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Log start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "Start time: ${START_TIME}"
echo ""

# Run the pipeline WITHOUT max_samples parameter to process all data
echo "Starting Gemini-based pipeline processing on FULL dataset..."
echo "Using audio-based verification for higher accuracy..."
echo "This will take significantly longer than text-only correction..."
echo ""

uv run python main.py \
    --config-path dataset_configs/vox_pipeline/granary/ \
    --config-name config_pl_gemini.yaml \
    data_dir="${DATA_DIR}" \
    output_dir="${OUTPUT_DIR}" \
    sdp_dir="${SDP_DIR}" \
    params.max_samples=${MAX_SAMPLES} \
    params.audio_channel=mic \
    params.save_disk_space=false \
    processors_to_run=all \
    2>&1 | tee ${OUTPUT_DIR}/pipeline_full.log

# Note: Remove MAX_SAMPLES variable and params.max_samples line above to process all data

# Log end time
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo ""
echo "========================================"
echo "Pipeline completed!"
echo "========================================"
echo "Start time: ${START_TIME}"
echo "End time: ${END_TIME}"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Gemini Audio Corrections Applied:"
echo "  - Direct audio verification"
echo "  - Menu item pronunciation matching"
echo "  - Polish model error corrections"
echo "  - Preposition recovery (z kurczakiem)"
echo "  - Punctuation normalization"
echo "  - Higher accuracy than text-only"
echo ""
echo "Processing Statistics:"
echo "  - Check ${OUTPUT_DIR}/pipeline_full.log for details"
echo "  - Final manifest: ${OUTPUT_DIR}/pl/manifest_32.json"
echo "========================================"