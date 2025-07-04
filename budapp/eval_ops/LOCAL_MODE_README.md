# Evaluation Data Sync - Local Mode

## Overview

The evaluation data sync workflow supports a local mode for development and testing purposes. When `eval_sync_local_mode` is set to `true` in the configuration, the system will load manifest and dataset files from local directories instead of fetching them from the cloud.

## Configuration

Set the following environment variable to enable local mode:

```bash
EVAL_SYNC_LOCAL_MODE=true
```

## Local File Structure

When local mode is enabled, the system expects the following file structure:

```
budapp/initializers/data/
├── eval_manifest.json                    # The manifest file
└── eval_datasets/                        # Dataset files matching the paths in manifest
    ├── opencompass/
    │   ├── gsm8k.jsonl
    │   └── humaneval.jsonl
    └── bud_custom/
        └── multilingual_qa.jsonl
```

## How It Works

1. **Manifest Loading**: Instead of fetching from `eval_manifest_url`, the system loads `budapp/initializers/data/eval_manifest.json`

2. **Dataset Loading**: Instead of downloading from URLs, the system:
   - Looks for dataset files in `budapp/initializers/data/eval_datasets/` + the dataset URL path
   - If a file exists, it copies it to the cache directory
   - If a file doesn't exist, it creates a dummy JSONL file with test data

3. **Bundle Downloads**: Bundle downloads are disabled in local mode. The system will always use individual dataset files.

4. **Checksum Verification**: 
   - In development environment (`env=dev`), checksum verification is skipped
   - In other environments, checksums are still verified even for local files

## Usage Example

1. Enable local mode in your `.env` file:
   ```
   EVAL_SYNC_LOCAL_MODE=true
   ```

2. Place your manifest file at:
   ```
   budapp/initializers/data/eval_manifest.json
   ```

3. Place dataset files in the appropriate subdirectories under:
   ```
   budapp/initializers/data/eval_datasets/
   ```

4. Run the sync workflow as usual. The system will automatically use local files.

## Benefits

- **Faster Development**: No need to download large datasets repeatedly
- **Offline Development**: Work without internet connectivity
- **Testing**: Easy to test with custom datasets and manifests
- **Debugging**: Simplified debugging of sync logic without network issues

## Limitations

- Bundle downloads are not supported in local mode
- All dataset files must be placed manually in the correct directory structure
- Authentication features are bypassed for local files