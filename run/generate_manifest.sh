#!/bin/bash

# Generate Initial Manifest for EJ AU Drive-thru Dataset
# This script creates the initial manifest from the hierarchical audio directory

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== EJ AU Dataset Manifest Generator ===${NC}"
echo ""

# Set paths
DATA_DIR="data/audio"
OUTPUT_MANIFEST="data/initial_manifest.json"
AUDIO_TYPE="both"  # Options: mic, spk, both

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${YELLOW}Error: Data directory not found at $DATA_DIR${NC}"
    echo "Please ensure your audio files are in: data/audio/{country}/{location}/{device_id}/..."
    exit 1
fi

# Count audio files
AUDIO_COUNT=$(find "$DATA_DIR" -name "*.wav" 2>/dev/null | wc -l)
echo -e "${GREEN}Found $AUDIO_COUNT WAV files in $DATA_DIR${NC}"

# Generate manifest
echo -e "${YELLOW}Generating manifest...${NC}"
python sdp/processors/create_initial_manifest.py \
    --data-dir "$DATA_DIR" \
    --output-manifest "$OUTPUT_MANIFEST" \
    --audio-type "$AUDIO_TYPE" \
    --preserve-structure

# Check if manifest was created
if [ -f "$OUTPUT_MANIFEST" ]; then
    MANIFEST_LINES=$(wc -l < "$OUTPUT_MANIFEST")
    echo -e "${GREEN}✓ Manifest created successfully with $MANIFEST_LINES entries${NC}"
    echo -e "${GREEN}  Location: $OUTPUT_MANIFEST${NC}"
    
    # Show statistics if available
    STATS_FILE="${OUTPUT_MANIFEST%.json}.stats.json"
    if [ -f "$STATS_FILE" ]; then
        echo ""
        echo -e "${GREEN}Dataset Statistics:${NC}"
        python -c "import json; stats=json.load(open('$STATS_FILE')); print(json.dumps(stats, indent=2))"
    fi
else
    echo -e "${YELLOW}Error: Failed to create manifest${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Next step: Run the pipeline with:${NC}"
echo "  ./run/run_pipeline.sh"