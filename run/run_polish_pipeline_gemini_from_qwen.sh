#!/bin/bash

# Polish Drive-thru Pipeline - Gemini from Qwen Base
# This script runs ONLY the Gemini-specific stages (24+) using manifest_23 from Qwen pipeline
# Requires: Qwen pipeline to have completed at least up to manifest_23.json

set -e

# Configuration
DATA_DIR="data/audio"
QWEN_OUTPUT_DIR="outputs/vox_pipeline_pl_qwen_full"  # Source for manifest_23 from FULL Qwen run
OUTPUT_DIR="outputs/vox_pipeline_pl_gemini_from_qwen"
SDP_DIR="$(pwd)"

echo "=========================================="
echo "Polish Pipeline - Gemini from Qwen Base (FULL DATASET)"
echo "=========================================="
echo "This runs ONLY Gemini audio correction stages (24+)"
echo "Using manifest_23 from: ${QWEN_OUTPUT_DIR}"
echo ""

# Check if GEMINI_API_KEY is set
if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY environment variable is not set"
    echo "Please export GEMINI_API_KEY=your_api_key"
    exit 1
fi

# Check if manifest_23 exists from Qwen pipeline
MANIFEST_23="${QWEN_OUTPUT_DIR}/pl/manifest_23.json"
if [ ! -f "$MANIFEST_23" ]; then
    echo "ERROR: Required manifest not found: $MANIFEST_23"
    echo "Please run Qwen FULL pipeline first with:"
    echo "  ./run/run_polish_pipeline_qwen_full.sh"
    exit 1
fi

echo "✓ Found manifest_23.json from Qwen pipeline"
echo "✓ GEMINI_API_KEY is set"
echo ""

# Count entries in manifest_23
ENTRY_COUNT=$(grep -c '"audio_filepath"' "$MANIFEST_23" || true)
echo "Processing $ENTRY_COUNT entries from Qwen manifest"
echo ""

# Estimate time based on entry count and rate limit (150 RPM for Tier 1)
ESTIMATED_TIME_MIN=$((ENTRY_COUNT / 145))  # At configured rate
ESTIMATED_TIME_REALISTIC=$((ENTRY_COUNT / 120))  # More realistic with processing overhead
echo "Estimated processing time: ${ESTIMATED_TIME_MIN}-${ESTIMATED_TIME_REALISTIC} minutes"
echo "Note: Using Gemini 2.5 Pro Tier 1 with 145 RPM (safety margin from 150 limit)"
echo "This only processes Gemini verification + punctuation restoration"
echo ""

# Check if output directory already has data - offer to resume
if [ -f "${OUTPUT_DIR}/pl/manifest_24.json" ]; then
    echo "WARNING: Found existing Gemini output at stage 24!"
    echo "File: ${OUTPUT_DIR}/pl/manifest_24.json exists"
    
    # Count how many entries were already processed
    PROCESSED_COUNT=$(grep -c '"gemini_confidence"' "${OUTPUT_DIR}/pl/manifest_24.json" 2>/dev/null || echo "0")
    echo "Already processed: ${PROCESSED_COUNT} entries with Gemini"
    echo ""
    echo "The pipeline will automatically RESUME from where it left off."
    echo "Cached entries will be skipped, only unprocessed entries will be sent to Gemini API."
    echo ""
    read -p "Continue with resume? (Y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        read -p "Do you want to start fresh instead? This will DELETE existing progress (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing existing output directory..."
            rm -rf ${OUTPUT_DIR}
        else
            echo "Aborting to preserve existing data."
            exit 1
        fi
    fi
elif [ -f "${OUTPUT_DIR}/pl/manifest_32.json" ]; then
    echo "WARNING: Output directory contains a complete pipeline run!"
    echo "File: ${OUTPUT_DIR}/pl/manifest_32.json exists"
    read -p "Do you want to re-run the entire pipeline? (y/N): " -n 1 -r
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
cat > ${OUTPUT_DIR}/monitor_gemini_progress.sh << 'EOF'
#!/bin/bash
# Monitor Gemini pipeline progress
OUTPUT_DIR="$1"
QWEN_DIR="$2"

while true; do
    clear
    echo "Gemini Pipeline Progress Monitor"
    echo "================================"
    echo ""
    
    # Show which manifest we're copying from
    if [ -f "${QWEN_DIR}/pl/manifest_23.json" ]; then
        QWEN_COUNT=$(grep -c '"audio_filepath"' "${QWEN_DIR}/pl/manifest_23.json" 2>/dev/null || echo "0")
        echo "Source (Qwen manifest_23): $QWEN_COUNT entries"
        echo ""
    fi
    
    # Check Gemini-specific stages
    echo "Gemini Processing Stages:"
    if [ -f "${OUTPUT_DIR}/pl/manifest_23.json" ]; then
        echo "✓ Stage 23: Copied from Qwen"
    else
        echo "  Stage 23: Copying from Qwen..."
    fi
    
    for i in {24..32}; do
        MANIFEST="${OUTPUT_DIR}/pl/manifest_${i}.json"
        if [ -f "$MANIFEST" ]; then
            COUNT=$(grep -c '"audio_filepath"' "$MANIFEST" 2>/dev/null || echo "0")
            case $i in
                24) echo "✓ Stage 24: Gemini audio verification - $COUNT entries" ;;
                25) echo "✓ Stage 25: Punctuation restoration (Qwen) - $COUNT entries" ;;
                26) echo "✓ Stage 26: Clean Qwen output - $COUNT entries" ;;
                27) echo "✓ Stage 27: Text normalization - $COUNT entries" ;;
                28) echo "✓ Stage 28: Drop fields - $COUNT entries" ;;
                29) echo "✓ Stage 29: Rename fields - $COUNT entries" ;;
                30) echo "✓ Stage 30: Add metadata - $COUNT entries" ;;
                31) echo "✓ Stage 31: Final field selection - $COUNT entries" ;;
                32) echo "✓ Stage 32: WebDataset conversion - $COUNT entries" ;;
            esac
        else
            case $i in
                24) echo "  Stage 24: Gemini audio verification - pending..." ;;
                25) echo "  Stage 25: Punctuation restoration - pending..." ;;
                26) echo "  Stage 26: Clean output - pending..." ;;
                27) echo "  Stage 27: Text normalization - pending..." ;;
                28) echo "  Stage 28: Drop fields - pending..." ;;
                29) echo "  Stage 29: Rename fields - pending..." ;;
                30) echo "  Stage 30: Add metadata - pending..." ;;
                31) echo "  Stage 31: Final selection - pending..." ;;
                32) echo "  Stage 32: WebDataset - pending..." ;;
            esac
            break
        fi
    done
    
    echo ""
    echo "Last update: $(date '+%H:%M:%S')"
    echo "Press Ctrl+C to stop monitoring"
    sleep 10
done
EOF
chmod +x ${OUTPUT_DIR}/monitor_gemini_progress.sh

echo "TIP: To monitor progress in another terminal, run:"
echo "  ${OUTPUT_DIR}/monitor_gemini_progress.sh ${OUTPUT_DIR} ${QWEN_OUTPUT_DIR}"
echo ""

# Log start time
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
START_EPOCH=$(date +%s)
echo "Start time: ${START_TIME}"
echo "=========================================="
echo ""

# Function to handle interruption
cleanup() {
    echo ""
    echo "=========================================="
    echo "Gemini pipeline interrupted!"
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

# Run only Gemini-specific stages
echo "Running Gemini audio verification on Qwen-processed data..."
echo "This will make API calls to Gemini for each audio segment..."
echo ""

uv run python main.py \
    --config-path dataset_configs/vox_pipeline/granary/ \
    --config-name config_pl_gemini_from_qwen.yaml \
    data_dir="${DATA_DIR}" \
    qwen_output_dir="${QWEN_OUTPUT_DIR}" \
    output_dir="${OUTPUT_DIR}" \
    sdp_dir="${SDP_DIR}" \
    params.audio_channel=mic \
    params.save_disk_space=false \
    processors_to_run=all \
    2>&1 | tee ${OUTPUT_DIR}/gemini_from_qwen.log

# Calculate duration
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
END_EPOCH=$(date +%s)
DURATION=$((END_EPOCH - START_EPOCH))
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))

echo ""
echo "=========================================="
echo "Gemini pipeline completed successfully!"
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
    
    # Compare with Qwen output
    if [ -f "${QWEN_OUTPUT_DIR}/pl/manifest_23.json" ]; then
        QWEN_COUNT=$(grep -c '"audio_filepath"' "${QWEN_OUTPUT_DIR}/pl/manifest_23.json" 2>/dev/null || echo "0")
        echo ""
        echo "Comparison:"
        echo "  - Started with: ${QWEN_COUNT} segments from Qwen"
        echo "  - Ended with: ${FINAL_COUNT} segments after Gemini"
    fi
else
    echo "WARNING: Final manifest not found!"
    echo "Check ${OUTPUT_DIR}/gemini_from_qwen.log for errors"
fi

echo ""
echo "Gemini Audio Corrections Applied:"
echo "  ✓ Listened to actual audio for verification"
echo "  ✓ Fixed menu item spelling based on pronunciation"
echo "  ✓ Added confidence scores for corrections"
echo "  ✓ Preserved Polish words (kurczak, ser, etc.)"
echo "  ✓ Only corrected actual misspellings (whooper → whopper)"
echo ""

# Check cache usage from the log
if [ -f "${OUTPUT_DIR}/gemini_from_qwen.log" ]; then
    CACHED_USED=$(grep "Cached results used:" "${OUTPUT_DIR}/gemini_from_qwen.log" | tail -1 | sed 's/.*Cached results used: //' | cut -d' ' -f1)
    if [ ! -z "$CACHED_USED" ] && [ "$CACHED_USED" -gt 0 ]; then
        echo "Cache Usage:"
        echo "  - Cached entries used: ${CACHED_USED}"
        echo "  - New API calls made: $((FINAL_COUNT - CACHED_USED))"
        echo ""
    fi
fi

echo "Processing included:"
echo "  - Gemini audio verification (Stage 24)"
echo "  - Qwen punctuation restoration (Stage 25)"
echo "  - Text normalization (Stages 26-27)"
echo "  - WebDataset creation (Stage 32)"
echo ""
echo "Next Steps:"
echo "  1. Review the log: less ${OUTPUT_DIR}/gemini_from_qwen.log"
echo "  2. Compare with Qwen: python scripts/compare_qwen_gemini_corrections.py"
echo "  3. Upload to HuggingFace: ./run/upload_webdataset_to_hf.sh ${OUTPUT_DIR}/pl/webdataset VoxAI/bk-pl-$(date +%Y%m%d)-gemini"
echo "=========================================="