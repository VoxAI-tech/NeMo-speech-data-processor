# Speech Data Processor (SDP) Toolkit

The Speech Data Processor (SDP) is a toolkit designed to simplify the processing of speech datasets. It minimizes the boilerplate code required and allows for easy sharing of processing steps. SDP's philosophy is to represent processing operations as 'processor' classes, which take in a path to a NeMo-style data manifest as input (or a path to the raw data directory if you do not have a NeMo-style manifest to start with), apply some processing to it, and then save the output manifest file.

## Vox Pipeline - Drive-Thru Audio Processing

This repository includes specialized pipelines for processing drive-thru audio data, particularly optimized for Polish language ASR using advanced error correction techniques.

### Available Pipelines

#### 1. **Qwen Pipeline** (Fast, Text-based Correction)
- Uses Qwen3-8B LLM for text-based menu item correction
- Fuzzy matching against menu vocabulary
- Punctuation and capitalization restoration
- No API rate limits

#### 2. **Gemini Pipeline** (Audio-based Verification)
- Uses Gemini 2.5 Pro to listen to actual audio
- Verifies and corrects menu item spellings based on pronunciation
- Higher accuracy but limited by API quotas (10,000 requests/day)
- Automatic caching and resume on rate limit interruptions
- Can reuse Qwen outputs to save 70% processing time

### Key Features

- **Polish ASR Optimization:** Specialized for voxai/whisper-large-v3-polish-ct2 model
- **Menu-Aware Correction:** Context-aware fixing of menu item transcriptions
- **Multi-Stage Processing:** 32+ processing stages from raw audio to WebDataset
- **Language Detection:** Automatic filtering of non-target language segments
- **Smart Segmentation:** VAD-based audio splitting with configurable thresholds
- **WebDataset Export:** Ready for training with tar archives

### Documentation

- **[Pipeline Architecture](docs/vox_pipeline.md):** Complete technical documentation of the 32-stage pipeline
  - Detailed configuration explanations for each processing stage
  - Input/output manifest transformations at each step
  - Parameter tuning guidelines and best practices
  - Troubleshooting common issues

- **[Future Enhancements](docs/proposal.md):** Research-backed improvements that can be added to further optimize the pipeline:
  - Cross-channel audio bleed reduction using spectral subtraction
  - Advanced dual-stage VAD for short utterances (0.5-3s)
  - Multi-pass processing architecture with iterative refinement
  - Active learning integration for sample selection
  - Whisper contextual biasing without full fine-tuning
  - Expected WER improvements: 25-40% reduction

## Features

- **Creating Manifests:** Generate manifests for your datasets.
- **Running ASR Inference:** Automatically run ASR inference to remove utterances where the reference text differs greatly from ASR predictions.
- **Text Transformations:** Apply text-based transformations to lines in the manifest.
- **Removing Inaccurate Transcripts:** Remove lines from the manifest which may contain inaccurate transcripts.
- **Custom Processors:** Write your own processor classes if the provided ones do not meet your needs.

## Installation

SDP is officially supported for Python 3.10, but might work for other versions.

1. Clone the repository:

```bash
   git clone https://github.com/VoxAI-Tech/NeMo-speech-data-processor.git
   cd NeMo-speech-data-processor
```
2. Install dependencies:
```bash
   uv sync
   pip install -r requirements/main.txt (TODO add dependencies to pyproject)
```

3. Optional: If you need to use ASR, NLP parts, or NeMo Text Processing, follow the NeMo installation instructions:
   - [NeMo Installation](https://github.com/NVIDIA/NeMo)

## Quick Start - Vox Pipeline

### Running the Polish Drive-Thru Pipeline

#### Option 1: Qwen Pipeline (Recommended for full processing)
```bash
# Process entire dataset with Qwen LLM correction
./run/run_polish_pipeline_qwen_full.sh

# Monitor progress in another terminal
tail -f outputs/vox_pipeline_pl_qwen_full/qwen_full.log
```

#### Option 2: Gemini Pipeline (For higher accuracy)
```bash
# First, set your Gemini API key
export GEMINI_API_KEY="your_api_key_here"

# Run Gemini pipeline (reuses Qwen outputs if available)
./run/run_polish_pipeline_gemini_from_qwen.sh

# Check progress
./run/check_gemini_progress.sh
```

### Pipeline Configuration

The pipelines are configured in `dataset_configs/vox_pipeline/granary/`:
- `config_pl_qwen.yaml` - Full Qwen pipeline with 32 processing stages
- `config_pl_gemini_from_qwen.yaml` - Gemini audio verification (stages 24-32)

For detailed configuration options and stage-by-stage explanations, see [Pipeline Architecture Documentation](docs/vox_pipeline.md).

### Processing Stages Overview

The pipeline consists of 32 stages organized into 9 phases:

1. **Audio Preparation** (Stages 0-4): Format standardization and metadata extraction
2. **Language Detection** (Stages 5-7): Filter for target language (Polish)
3. **Initial Transcription** (Stages 8-10): Whisper ASR with segment boundaries
4. **Segment Processing** (Stages 11-15): Split into individual utterances
5. **Refined Transcription** (Stage 16-18): Second-pass ASR on segments
6. **Quality Filtering** (Stages 19-22): Remove hallucinations and empty text
7. **Text Correction** (Stages 23-24): Menu-aware fuzzy matching + Gemini verification
8. **Text Enhancement** (Stages 25-27): Punctuation restoration and normalization
9. **Final Processing** (Stages 28-32): WebDataset conversion

See [docs/vox_pipeline.md](docs/vox_pipeline.md) for complete technical details of each stage.

### Upload to Hugging Face

```bash
# Upload processed dataset
./run/upload_webdataset_to_hf.sh outputs/vox_pipeline_pl_qwen_full/pl/webdataset VoxAI/bk-pl-$(date +%Y%m%d)
```

### Advanced Usage

#### Custom Menu Vocabulary
Edit `dataset_configs/vox_pipeline/granary/bk_menu_vocabulary.json` to add restaurant-specific menu items:
```json
{
  "whopper": ["whooper", "whooper", "woper"],
  "kurczak": ["kurczok", "kurczag"],
  "frytki": ["frytke", "fritki"]
}
```

#### Adjusting Processing Parameters
Edit `dataset_configs/vox_pipeline/granary/config_pl_qwen.yaml`:
```yaml
params:
  min_audio_duration: 0.5  # Minimum segment duration in seconds
  max_audio_duration: 30   # Maximum segment duration
  min_audio_lid_probability: 0.7  # Language detection threshold
```

For implementing advanced features like cross-channel processing or improved VAD, refer to [Enhancement Proposals](docs/proposal.md).

## Example:
1. In this example we will load librispeech using SDP.
   * For downloading all available data - replace config.yaml with all.yaml
   * For mini dataset - replace with mini.yaml.
```bash
    python NeMo-speech-data-processor/main.py \
    --config-path="dataset_configs/english/librispeech" \
    --config-name="config.yaml" \
    processors_to_run="0:" \
    workspace_dir="librispeech_data_dir"
```
## Usage

1. Create a Configuration YAML File:

   Here is a simplified example of a `config.yaml` file:

   ```yaml
   processors:
     - _target_: sdp.processors.CreateInitialManifestMCV
       output_manifest_file: "${data_split}_initial_manifest.json"
       language_id: es
     - _target_: sdp.processors.ASRInference
       pretrained_model: "stt_es_quartznet15x5"
     - _target_: sdp.processors.SubRegex
       regex_params_list:
         - {"pattern": "¡", "repl": "."}
         - {"pattern": "ó", "repl": "o"}
       test_cases:
         - {input: {text: "hey!"}, output: {text: "hey."}}
     - _target_: sdp.processors.DropNonAlphabet
       alphabet: "abcdefghijklmnopqrstuvwxyzáéiñóúüABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÑÓÚÜ"
       test_cases:
         - {input: {text: "test Тест ¡"}, output: null}
         - {input: {text: "test"}, output: {text: "test"}}
     - _target_: sdp.processors.KeepOnlySpecifiedFields
       output_manifest_file: "${data_split}_final_manifest.json"
       fields_to_keep:
         - "audio_filepath"
         - "text"
         - "duration"
   ```

2. Run the Processor:

   Use the following command to process your dataset:

```bash
   python <SDP_ROOT>/main.py \
     --config-path="dataset_configs/<lang>/<dataset>/" \
     --config-name="config.yaml" \
     processors_to_run="all" \
     data_split="train" \
     workspace_dir="<dir_to_store_processed_data>"
```

![SDP overview](https://github.com/NVIDIA/NeMo/releases/download/v1.17.0/sdp_overview_diagram.png)

To learn more about SDP, have a look at our [documentation](https://nvidia.github.io/NeMo-speech-data-processor/).


## Contributing
We welcome community contributions! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for the process.
