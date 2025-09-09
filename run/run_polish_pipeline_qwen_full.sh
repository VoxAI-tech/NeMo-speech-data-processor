#!/bin/bash

# Polish Drive-thru Pipeline with Qwen LLM Menu Correction
# Full dataset processing version
# Uses vLLM library directly - no server needed

set -e

# Configuration
DATA_DIR="data/audio"
OUTPUT_DIR="outputs/vox_pipeline_pl_qwen_full"
SDP_DIR="$(pwd)"

echo "=========================================="
echo "Polish Pipeline - Qwen LLM Version (FULL DATASET)"
echo "=========================================="
echo "Model: VoxAI/whisper-large-v3-polish-ct2"
echo "Correction: Qwen3-8B (text-based via vLLM)"
echo "Data: ${DATA_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Processing: FULL DATASET (7,391 files)"
echo "=========================================="
echo ""
echo "WARNING: This will process the ENTIRE dataset"
echo "Estimated time: 4-6 hours"
echo "Estimated disk space needed: ~5GB"
echo ""

# Check if output directory already has data
if [ -f "${OUTPUT_DIR}/pl/manifest_32.json" ]; then
    echo "WARNING: Output directory already contains processed data!"
    echo "File: ${OUTPUT_DIR}/pl/manifest_32.json exists"
    read -p "Do you want to overwrite? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting to preserve existing data."
        exit 1
    fi
    echo "Removing existing output directory..."
    rm -rf ${OUTPUT_DIR}
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Create a simple progress monitor script
cat > ${OUTPUT_DIR}/monitor_progress.sh << 'EOF'
#!/bin/bash
# Monitor pipeline progress by checking manifest files
OUTPUT_DIR="$1"
while true; do
    clear
    echo "Pipeline Progress Monitor"
    echo "========================="
    echo ""
    for i in {00..32}; do
        MANIFEST="${OUTPUT_DIR}/pl/manifest_${i}.json"
        if [ -f "$MANIFEST" ]; then
            COUNT=$(grep -c '"audio_filepath"' "$MANIFEST" 2>/dev/null || echo "0")
            echo "✓ Stage $i: $COUNT entries"
        else
            echo "  Stage $i: pending..."
            break
        fi
    done
    echo ""
    echo "Last update: $(date '+%H:%M:%S')"
    echo "Press Ctrl+C to stop monitoring"
    sleep 10
done
EOF
chmod +x ${OUTPUT_DIR}/monitor_progress.sh

echo "TIP: To monitor progress in another terminal, run:"
echo "  ${OUTPUT_DIR}/monitor_progress.sh ${OUTPUT_DIR}"
echo ""

# Log start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
START_EPOCH=$(date +%s)
echo "Start time: ${START_TIME}"
echo ""

# Function to handle interruption
cleanup() {
    echo ""
    echo "=========================================="
    echo "Pipeline interrupted!"
    echo "=========================================="
    END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
    END_EPOCH=$(date +%s)
    DURATION=$((END_EPOCH - START_EPOCH))
    HOURS=$((DURATION / 3600))
    MINUTES=$(( (DURATION % 3600) / 60 ))
    echo "Start time: ${START_TIME}"
    echo "End time: ${END_TIME}"
    echo "Duration: ${HOURS}h ${MINUTES}m"
    echo ""
    echo "Partial results saved to: ${OUTPUT_DIR}"
    echo "Check the last manifest file to see progress."
    exit 1
}

# Set up trap for Ctrl+C
trap cleanup INT

# Run the pipeline WITHOUT max_samples parameter to process all data
echo "Starting Qwen-based pipeline processing on FULL dataset..."
echo "vLLM will be initialized directly by the processor - no server needed."
echo "This will take several hours..."
echo ""

uv run python main.py \
    --config-path dataset_configs/vox_pipeline/granary/ \
    --config-name config_pl_qwen.yaml \
    data_dir="${DATA_DIR}" \
    output_dir="${OUTPUT_DIR}" \
    sdp_dir="${SDP_DIR}" \
    params.audio_channel=mic \
    params.save_disk_space=false \
    processors_to_run=all \
    2>&1 | tee ${OUTPUT_DIR}/pipeline_full.log

# Calculate duration
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
END_EPOCH=$(date +%s)
DURATION=$((END_EPOCH - START_EPOCH))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Start time: ${START_TIME}"
echo "End time: ${END_TIME}"
echo "Duration: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""

# Check final output
if [ -f "${OUTPUT_DIR}/pl/manifest_32.json" ]; then
    FINAL_COUNT=$(grep -c '"audio_filepath"' "${OUTPUT_DIR}/pl/manifest_32.json" 2>/dev/null || echo "0")
    echo "Final dataset: ${FINAL_COUNT} segments processed"
    
    # Check WebDataset output
    if [ -d "${OUTPUT_DIR}/pl/webdataset" ]; then
        TAR_COUNT=$(ls ${OUTPUT_DIR}/pl/webdataset/train/*.tar 2>/dev/null | wc -l)
        echo "WebDataset: ${TAR_COUNT} tar shards created"
    fi
else
    echo "WARNING: Final manifest not found!"
    echo "Check ${OUTPUT_DIR}/pipeline_full.log for errors"
fi

echo ""
echo "Qwen LLM Corrections Applied:"
echo "  ✓ Menu item spelling fixes (whooper → whopper)"
echo "  ✓ Polish model error corrections"
echo "  ✓ Preposition recovery (kurczakiem → z kurczakiem)"
echo "  ✓ Punctuation and capitalization restoration"
echo "  ✓ Filler word preservation"
echo ""
echo "Next Steps:"
echo "  1. Review the log: less ${OUTPUT_DIR}/pipeline_full.log"
echo "  2. Upload to HuggingFace: ./run/upload_webdataset_to_hf.sh ${OUTPUT_DIR}/pl/webdataset VoxAI/bk-pl-$(date +%Y%m%d)"
echo "  3. Or run Gemini verification: ./run/run_polish_pipeline_gemini_from_qwen.sh"
echo "=========================================="