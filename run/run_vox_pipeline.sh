#!/bin/bash

# Run script for Vox Pipeline (Drive-thru Audio Processing)
# This script processes El Jannah Australia drive-thru audio conversations

# Set the paths
SDP_DIR="/home/razhan/NeMo-speech-data-processor"
DATA_DIR="${SDP_DIR}/data/audio"
OUTPUT_DIR="${SDP_DIR}/outputs/vox_pipeline_$(date +%Y%m%d_%H%M%S)"

# GPU Configuration for vLLM
export CUDA_VISIBLE_DEVICES=1  # Use GPU 1 for vLLM inference
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings

# Configuration parameters
AUDIO_CHANNEL="mic"  # Change to "spk" for employee audio
MAX_SAMPLES=10       # Testing with 10 samples (change to -1 for all samples)
CONFIG_NAME="config.yaml"  # Use config.yaml for full pipeline, config_simple.yaml for testing

# Tar creation parameters
CREATE_TAR=false     # Disabled for testing (set to true for production)
TAR_CHUNK_SIZE=1000  # Number of files per tar chunk
TAR_NUM_WORKERS=8    # Number of parallel workers for tar creation

# Advanced processing parameters
NUM_WORKERS=8        # Reduced for testing (increase to 16+ for production)
BATCH_SIZE=16        # Reduced for testing (increase to 32+ for production)
USE_DASK=true        # Use Dask for distributed processing

echo "======================================"
echo "Vox Pipeline - Drive-thru Audio Processing"
echo "======================================"
echo "SDP Directory: ${SDP_DIR}"
echo "Data Directory: ${DATA_DIR}"
echo "Output Directory: ${OUTPUT_DIR}"
echo "Config: ${CONFIG_NAME}"
echo "Audio Channel: ${AUDIO_CHANNEL}"
echo "Max Samples: ${MAX_SAMPLES}"
echo "GPU Device: ${CUDA_VISIBLE_DEVICES}"
echo "Create Tar: ${CREATE_TAR}"
echo "Workers: ${NUM_WORKERS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "======================================"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Build the command with all parameters
CMD="uv run python ${SDP_DIR}/main.py \
    --config-path ${SDP_DIR}/dataset_configs/vox_pipeline/granary/ \
    --config-name ${CONFIG_NAME} \
    data_dir=${DATA_DIR} \
    output_dir=${OUTPUT_DIR} \
    sdp_dir=${SDP_DIR} \
    params.audio_channel=${AUDIO_CHANNEL} \
    params.max_samples=${MAX_SAMPLES} \
    use_dask=${USE_DASK}"

# Add tar creation parameters if enabled
if [ "${CREATE_TAR}" = true ]; then
    CMD="${CMD} \
    params.convert_to_audio_tarred_dataset.should_run=true"
fi

echo "Running command:"
echo "${CMD}"
echo "======================================"

# Run the pipeline
eval ${CMD}

echo "======================================"
echo "Pipeline execution completed!"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""

# Display processing summary
echo "Processing Summary:"
echo "-------------------"

# Check for final manifest
FINAL_MANIFEST="${OUTPUT_DIR}/en/manifest_30.json"
if [ -f "${FINAL_MANIFEST}" ]; then
    ENTRY_COUNT=$(wc -l < "${FINAL_MANIFEST}")
    echo "✓ Final manifest created with ${ENTRY_COUNT} entries"
    echo "  Location: ${FINAL_MANIFEST}"
else
    # Check for last available manifest
    LAST_MANIFEST=$(ls -1 ${OUTPUT_DIR}/en/manifest_*.json 2>/dev/null | sort -V | tail -1)
    if [ -n "${LAST_MANIFEST}" ]; then
        STAGE=$(basename "${LAST_MANIFEST}" | sed 's/manifest_\([0-9]*\).*/\1/')
        ENTRY_COUNT=$(wc -l < "${LAST_MANIFEST}")
        echo "⚠ Pipeline stopped at stage ${STAGE}"
        echo "  Last manifest: ${LAST_MANIFEST}"
        echo "  Entries: ${ENTRY_COUNT}"
    else
        echo "✗ No manifest files found"
    fi
fi

# Display tar creation results if enabled
if [ "${CREATE_TAR}" = true ]; then
    echo ""
    echo "Tarred Datasets:"
    echo "----------------"
    if [ -d "${OUTPUT_DIR}/tarred_audio" ]; then
        TAR_COUNT=$(find ${OUTPUT_DIR}/tarred_audio -name "*.tar" -type f 2>/dev/null | wc -l)
        if [ ${TAR_COUNT} -gt 0 ]; then
            echo "✓ Created ${TAR_COUNT} tar files"
            find ${OUTPUT_DIR}/tarred_audio -name "*.tar" -type f | head -5
            [ ${TAR_COUNT} -gt 5 ] && echo "  ... and $((TAR_COUNT - 5)) more"
        else
            echo "✗ No tar files created"
        fi
    else
        echo "✗ Tarred audio directory not found"
    fi
fi

# Check for any errors
ERROR_LOG="${OUTPUT_DIR}/error.log"
if [ -f "${ERROR_LOG}" ]; then
    echo ""
    echo "⚠ Errors detected. Check: ${ERROR_LOG}"
fi

echo ""
echo "======================================" 