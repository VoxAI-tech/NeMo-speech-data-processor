#!/usr/bin/env -S uv run python
"""
CLI tool to upload datasets to HuggingFace Hub in webdataset structure.
Searches for wav/json pairs, optionally shuffles them, and distributes them into tar files for upload to HF Hub.

To properly configure dataset subsets and splits in the README.md frontmatter, use this structure:

---
configs:
- config_name: default
  data_files:
  - split: train
    path: "train/*.tar"
  - split: validation
    path: "validation/*.tar"
  - split: test
    path: "test/*.tar"
---

For multiple subsets, each with their own splits:
---
configs:
- config_name: subset1
  data_files:
  - split: train
    path: "subset1/train/*.tar"
  - split: validation
    path: "subset1/validation/*.tar"
- config_name: subset2
  data_files:
  - split: train
    path: "subset2/train/*.tar"
  - split: validation
    path: "subset2/validation/*.tar"
---

Then load with: load_dataset("repo_id", name="subset_name", split="split_name")
"""

import json
import pathlib
import tempfile
import time

from typing import List

import click
import yaml

from huggingface_hub import HfApi, create_repo


def upload_to_hf_hub(
    tar_files: List[pathlib.Path],
    repo_id: str,
    token: str = None,
    subset: str = None,
    split: str = "train",
) -> None:
    """Upload tar files to HuggingFace Hub."""
    api = HfApi(token=token)

    # Create repository if it doesn't exist
    try:
        repo_url = create_repo(
            repo_id, token=token, private=True, exist_ok=True, repo_type="dataset"
        )
        print(f"Repository {repo_id} ready at {repo_url}")
        # Wait after repository creation to avoid race conditions
        print("Waiting 5 seconds after repository creation...")
        time.sleep(5)
    except Exception as e:
        print(f"Warning: Could not create/access repository: {e}")
        # Try to continue anyway in case repo already exists

    # Upload each tar file
    for tar_file in tar_files:
        try:
            # Flatten directory structure - use just the filename since they're uniquely named
            if subset:
                path_in_repo = f"{subset}/{split}/{tar_file.name}"
            else:
                path_in_repo = f"{split}/{tar_file.name}"

            api.upload_file(
                path_or_fileobj=str(tar_file),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                token=token,
                repo_type="dataset",
            )
            print(f"Uploaded {tar_file.name} to {path_in_repo}")
        except Exception as e:
            print(f"Error uploading {tar_file.name}: {e}")


def create_readme_frontmatter(
    repo_id: str, token: str = None, subset: str = None, split: str = "train"
) -> None:
    """Create or update README.md with dataset frontmatter, merging configurations."""
    api = HfApi(token=token)

    # Prepare new configuration
    if subset:
        path_pattern = f"{subset}/{split}/*.tar"
        config_name = subset
    else:
        path_pattern = f"{split}/*.tar"
        config_name = "default"

    new_config = {
        "config_name": config_name,
        "data_files": [{"split": split, "path": path_pattern}],
    }

    # Try to get existing README and parse frontmatter
    existing_configs = []
    readme_content = "# Dataset\n\nThis dataset contains audio files organized in webdataset format.\n"

    try:
        existing_readme = api.hf_hub_download(
            repo_id=repo_id, filename="README.md", repo_type="dataset", token=token
        )
        with open(existing_readme, "r") as f:
            content = f.read()

        # Parse existing YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter_yaml = parts[1].strip()
                    frontmatter_data = yaml.safe_load(frontmatter_yaml)
                    if frontmatter_data and "configs" in frontmatter_data:
                        existing_configs = frontmatter_data["configs"]
                    readme_content = parts[2].strip()
                except yaml.YAMLError:
                    pass  # Invalid YAML, start fresh

    except:
        pass  # No existing README

    # Merge configurations - update existing or add new
    merged_configs = []
    config_updated = False

    for config in existing_configs:
        if config.get("config_name") == config_name:
            # Update existing config
            merged_configs.append(new_config)
            config_updated = True
        else:
            # Keep existing config
            merged_configs.append(config)

    # Add new config if not found
    if not config_updated:
        merged_configs.append(new_config)

    # Generate new frontmatter
    frontmatter_data = {"configs": merged_configs}
    frontmatter_yaml = yaml.dump(frontmatter_data, default_flow_style=False)

    full_content = f"---\n{frontmatter_yaml}---\n\n{readme_content}"

    try:
        # Upload updated README
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(full_content)
            temp_readme = f.name

        api.upload_file(
            path_or_fileobj=temp_readme,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
            repo_type="dataset",
        )

        pathlib.Path(temp_readme).unlink()  # Clean up temp file
        print(f"Updated README.md with merged frontmatter for {config_name}/{split}")

    except Exception as e:
        print(f"Warning: Could not update README.md: {e}")


@click.command()
@click.argument("tar_dir", type=click.Path(exists=True, path_type=pathlib.Path))
@click.argument("repo_id", type=str)
@click.option("--token", type=str, help="HuggingFace token (optional if logged in)")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Don't upload to hub, just show what would be uploaded",
)
@click.option("--subset", type=str, help="Subset name to organize files under")
@click.option("--split", type=str, default="train", help="Split name for the dataset")
@click.option(
    "--add-duration-postfix",
    is_flag=True,
    help="Add duration postfix from metadata JSON file to repo name (e.g., '-5h')",
)
def main(
    tar_dir: pathlib.Path,
    repo_id: str,
    token: str,
    dry_run: bool,
    subset: str,
    split: str,
    add_duration_postfix: bool,
):
    """
    Upload existing WebDataset tar files to HuggingFace Hub.

    TAR_DIR should contain .tar files created by create_ws_tars.py.
    The script will upload all tar files in the directory to HF Hub.

    Example:
        python push_webdataset_to_hf.py /path/to/tar/files my-org/my-dataset
        python push_webdataset_to_hf.py /path/to/tar/files my-org/my-dataset --subset music --split train
    """

    # Find all tar files in the directory (including subdirectories)
    print(f"Searching for tar files in {tar_dir}...")
    tar_files = list(tar_dir.rglob("*.tar"))

    if not tar_files:
        raise click.ClickException(f"No .tar files found in {tar_dir}")

    # Check that all tar filenames are unique (to prevent conflicts when flattening)
    filenames = [tar_file.name for tar_file in tar_files]
    duplicates = [name for name in filenames if filenames.count(name) > 1]
    if duplicates:
        raise click.ClickException(
            f"Duplicate tar filenames found: {set(duplicates)}. Use unique --tar-prefix values to avoid conflicts."
        )

    print(f"Found {len(tar_files)} tar files with unique names")

    # Modify repo_id with duration postfix if requested
    final_repo_id = repo_id
    if add_duration_postfix:
        # Look for metadata JSON files (e.g., *_metadata.json)
        metadata_files = list(tar_dir.rglob("*_metadata.json"))
        if metadata_files:
            metadata_file = metadata_files[0]  # Use first one found
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                duration_hours = metadata.get("duration_hours", 0)
                if duration_hours > 0:
                    # Round up to nearest hour
                    import math

                    rounded_hours = math.ceil(duration_hours)
                    final_repo_id = f"{repo_id}-{rounded_hours}h"
                    print(f"Added duration postfix: {repo_id} -> {final_repo_id}")
                else:
                    print(
                        "Warning: Duration not found in metadata, using original repo name"
                    )
            except Exception as e:
                print(f"Warning: Could not read metadata file {metadata_file}: {e}")
                print("Using original repo name")
        else:
            print("Warning: No metadata JSON file found, using original repo name")

    if dry_run:
        print(
            f"\nDry run - would upload {len(tar_files)} tar files to {final_repo_id}:"
        )
        for tar_file in tar_files:
            # Flatten directory structure - use just the filename since they're uniquely named
            if subset:
                path_in_repo = f"{subset}/{split}/{tar_file.name}"
            else:
                path_in_repo = f"{split}/{tar_file.name}"
            print(f"  {tar_file.name} -> {path_in_repo}")
        return

    # Upload to HuggingFace Hub
    print("Uploading to HuggingFace Hub...")

    upload_to_hf_hub(tar_files, final_repo_id, token, subset, split)

    # Create/update README with frontmatter
    create_readme_frontmatter(final_repo_id, token, subset, split)

    print(f"\nDataset uploaded to {final_repo_id}")
    print(f"Tar files uploaded: {len(tar_files)}")


if __name__ == "__main__":
    main()
