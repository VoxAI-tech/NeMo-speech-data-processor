#!/usr/bin/env -S uv run python
"""
Upload WebDataset TAR archives to HuggingFace Hub.
"""

import json
import pathlib
import tempfile
import time
import click
import yaml
from huggingface_hub import HfApi, create_repo


def upload_webdataset_to_hf(
    webdataset_dir: pathlib.Path,
    repo_id: str,
    token: str = None,
) -> None:
    """Upload WebDataset TAR files to HuggingFace Hub."""
    api = HfApi(token=token)
    
    # Create repository if it doesn't exist
    try:
        repo_url = create_repo(
            repo_id, token=token, private=True, exist_ok=True, repo_type="dataset"
        )
        print(f"Repository {repo_id} ready at {repo_url}")
        print("Waiting 5 seconds after repository creation...")
        time.sleep(5)
    except Exception as e:
        print(f"Warning: Could not create/access repository: {e}")
    
    # Find all TAR files in train directory
    train_dir = webdataset_dir / "train"
    tar_files = list(train_dir.glob("*.tar"))
    
    print(f"Found {len(tar_files)} TAR files to upload")
    
    # Upload each TAR file
    uploaded_count = 0
    for tar_file in tar_files:
        try:
            # Upload to train/ directory in repo
            path_in_repo = f"train/{tar_file.name}"
            
            api.upload_file(
                path_or_fileobj=str(tar_file),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
                repo_type="dataset",
            )
            uploaded_count += 1
            print(f"Uploaded {tar_file.name} to {path_in_repo}")
            
        except Exception as e:
            print(f"Error uploading {tar_file.name}: {e}")
    
    # Upload metadata if exists
    metadata_file = webdataset_dir / "dataset_metadata.json"
    if metadata_file.exists():
        try:
            # Don't put in data folder - causes loading issues
            api.upload_file(
                path_or_fileobj=str(metadata_file),
                path_in_repo="dataset_info.json",
                repo_id=repo_id,
                token=token,
                repo_type="dataset",
            )
            print("Uploaded dataset_info.json")
        except Exception as e:
            print(f"Error uploading metadata: {e}")
    
    print(f"Successfully uploaded {uploaded_count} TAR files")


def create_webdataset_card(
    webdataset_dir: pathlib.Path,
    repo_id: str,
    token: str = None,
) -> None:
    """Create dataset card for WebDataset."""
    api = HfApi(token=token)
    
    # Read metadata if available
    metadata = {}
    metadata_file = webdataset_dir / "dataset_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    
    readme_content = f"""---
task_categories:
- automatic-speech-recognition
- text-to-speech
language:
- en
tags:
- audio
- speech
- webdataset
- drive-thru
- conversational
pretty_name: Drive-Thru Speech WebDataset
size_categories:
- n<1K
license: apache-2.0
configs:
- config_name: default
  data_files:
  - split: train
    path: "train/*.tar"
---

# Drive-Thru Speech WebDataset

This dataset is in WebDataset format, optimized for efficient streaming and loading with PyTorch DataLoaders.

## Dataset Statistics

- **Format**: WebDataset (TAR archives)
- **Total shards**: {metadata.get('total_shards', 'N/A')}
- **Total samples**: {metadata.get('total_samples', 'N/A')}
- **Total duration**: {metadata.get('total_duration_hours', 0):.2f} hours
- **Audio types**: {metadata.get('audio_types', {})}

## WebDataset Structure

Each TAR archive contains:
- `.wav` files: 16kHz mono audio segments
- `.json` files: Metadata and transcriptions

Files are paired by session and segment ID (e.g., `4ba42c95-b899-4bf1-8042-bf532e66e6b7_segment_0001.wav` and `4ba42c95-b899-4bf1-8042-bf532e66e6b7_segment_0001.json`).

## Usage

### With HuggingFace Datasets

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{repo_id}", streaming=True)

# Iterate through samples
for sample in dataset['train']:
    audio = sample['audio']
    text = sample['text']
```

### With WebDataset Library

```python
import webdataset as wds
from torch.utils.data import DataLoader

# Create WebDataset
url = "https://huggingface.co/datasets/{repo_id}/resolve/main/train/shard_{{000000..{metadata.get('total_shards', 1)-1:06d}}}.tar"
dataset = wds.WebDataset(url).decode()

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)

# Iterate through batches
for batch in dataloader:
    wav_data = batch['wav']
    metadata = batch['json']
```

### Direct Streaming

```python
import webdataset as wds

# Stream directly from HuggingFace
url = "pipe:curl -s -L https://huggingface.co/datasets/{repo_id}/resolve/main/train/shard_000000.tar"
dataset = wds.WebDataset(url).decode()

for sample in dataset:
    audio = sample['wav']
    text = json.loads(sample['json'])['text']
```

## Data Fields

Each JSON metadata file contains:

- `text`: Transcription with corrections
- `transcription`: Alternative transcription key
- `segment_id`: Segment identifier
- `duration`: Audio duration in seconds
- `offset`: Original offset in source audio
- `session_id`: Recording session ID
- `device_id`: Recording device ID
- `audio_type`/`speaker`: Speaker type (customer/employee)
- `confidence`: Transcription confidence score
- `audio_bleeding`: Whether audio bleeding was detected
- `correction_source`: Source of transcription correction
- `language`: Language code

## Processing Pipeline

1. **Segmentation**: Split into speech segments
2. **Transcription**: High-quality ASR transcription
3. **Menu-aware Correction**: Context-aware corrections for menu items
4. **Cross-channel Validation**: Enhanced validation using dual-channel audio
5. **WebDataset Creation**: Packed into TAR archives for efficient streaming

## Performance

WebDataset format provides:
- **Efficient I/O**: Sequential reads from TAR archives
- **Streaming**: Direct streaming from cloud storage
- **Sharding**: Distributed training support
- **Shuffling**: Efficient approximate shuffling

## License

Apache 2.0

## Acknowledgments

Processed using NeMo Speech Data Processor with enhanced cross-channel validation and menu-aware correction.
"""
    
    try:
        # Upload README
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(readme_content)
            temp_readme = f.name
        
        api.upload_file(
            path_or_fileobj=temp_readme,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
            repo_type="dataset",
        )
        
        pathlib.Path(temp_readme).unlink()
        print("Created and uploaded dataset card (README.md)")
        
    except Exception as e:
        print(f"Warning: Could not create dataset card: {e}")


@click.command()
@click.argument("webdataset_dir", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("repo_id", type=str)
@click.option("--token", type=str, help="HuggingFace token (optional if logged in)")
@click.option("--dry-run", is_flag=True, help="Show what would be uploaded without uploading")
def main(
    webdataset_dir: pathlib.Path,
    repo_id: str,
    token: str,
    dry_run: bool,
):
    """
    Upload WebDataset TAR archives to HuggingFace Hub.
    
    WEBDATASET_DIR should contain the webdataset with train/ subdirectory containing TAR files.
    
    Example:
        python push_webdataset_to_hf.py outputs/webdataset my-org/my-dataset
    """
    
    # Check for TAR files
    train_dir = webdataset_dir / "train"
    if not train_dir.exists():
        raise click.ClickException(f"Train directory not found: {train_dir}")
    
    tar_files = list(train_dir.glob("*.tar"))
    if not tar_files:
        raise click.ClickException(f"No TAR files found in {train_dir}")
    
    print(f"Found {len(tar_files)} TAR files in {train_dir}")
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in tar_files)
    print(f"Total size: {total_size / (1024**2):.2f} MB")
    
    if dry_run:
        print(f"\nDry run - would upload to {repo_id}:")
        for tar_file in tar_files:
            print(f"  train/{tar_file.name} ({tar_file.stat().st_size / (1024**2):.2f} MB)")
        return
    
    # Upload to HuggingFace Hub
    print(f"\nUploading to HuggingFace Hub: {repo_id}")
    
    upload_webdataset_to_hf(webdataset_dir, repo_id, token)
    create_webdataset_card(webdataset_dir, repo_id, token)
    
    print(f"\nDataset uploaded to {repo_id}")
    print(f"View at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()