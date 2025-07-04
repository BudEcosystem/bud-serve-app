"""Service for syncing evaluation datasets from cloud repository."""

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

import aiohttp
from sqlalchemy.orm import Session

from ..commons import logging
from ..commons.config import app_settings
from ..commons.database import SessionLocal, engine
from .manifest_schemas import Dataset, EvalDataManifest


logger = logging.get_logger(__name__)


class EvalDataSyncService:
    """Service for managing evaluation dataset synchronization."""

    def __init__(self, base_cache_dir: str = "/tmp/eval_datasets"):
        """Initialize the sync service.

        Args:
            base_cache_dir: Base directory for caching downloaded datasets
        """
        self.base_cache_dir = Path(base_cache_dir)
        self.base_cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_manifest(self, manifest_url: str) -> EvalDataManifest:
        """Fetch and parse the manifest from the cloud repository or local file.

        Args:
            manifest_url: URL of the manifest file or local file path in local mode

        Returns:
            Parsed manifest object
        """
        # Check if local mode is enabled
        if app_settings.eval_sync_local_mode:
            # In local mode, treat manifest_url as a filename in the data directory
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_file_path)
            local_manifest_path = os.path.join(parent_dir, "initializers", "data", "eval_manifest.json")

            logger.info(f"Local mode enabled, loading manifest from: {local_manifest_path}")

            try:
                with open(local_manifest_path, "r") as f:
                    manifest_data = json.load(f)
                    return EvalDataManifest(**manifest_data)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Local manifest file not found: {local_manifest_path}") from e
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in local manifest file: {local_manifest_path}") from e

        # Cloud mode - fetch from URL
        async with aiohttp.ClientSession() as session, session.get(manifest_url) as response:
            response.raise_for_status()
            manifest_data = await response.json()
            return EvalDataManifest(**manifest_data)

    def get_current_version(self, db: Session) -> Optional[str]:
        """Get the current version from the database.

        Args:
            db: Database session

        Returns:
            Current version string or None if not set
        """
        from sqlalchemy import select

        from .models import EvalSyncState

        # Query for the most recent successful sync
        stmt = (
            select(EvalSyncState)
            .where(EvalSyncState.sync_status == "completed")
            .order_by(EvalSyncState.sync_timestamp.desc())
            .limit(1)
        )

        result = db.execute(stmt)
        sync_state = result.scalar_one_or_none()

        if sync_state:
            logger.info(f"Current sync version: {sync_state.manifest_version}")
            return sync_state.manifest_version

        logger.info("No previous sync found")
        return None

    async def download_dataset(self, dataset: Dataset, base_url: str, auth_token: Optional[str] = None) -> Path:
        """Download a dataset file or copy from local directory.

        Args:
            dataset: Dataset information
            base_url: Base URL for downloads
            auth_token: Optional authentication token

        Returns:
            Path to downloaded file
        """
        cache_path = self.base_cache_dir / dataset.url
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if local mode is enabled
        if app_settings.eval_sync_local_mode:
            # In local mode, look for dataset files in the data directory
            current_file_path = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_file_path)
            local_dataset_path = os.path.join(parent_dir, "initializers", "data", "eval_datasets", dataset.url)

            logger.info(f"Local mode enabled, looking for dataset at: {local_dataset_path}")

            if os.path.exists(local_dataset_path):
                # Copy the local file to cache directory
                shutil.copy2(local_dataset_path, cache_path)
                logger.info(f"Copied local dataset {dataset.id} to cache")

                # Skip checksum verification in local mode for development
                if app_settings.env == "dev":
                    logger.warning(f"Skipping checksum verification for {dataset.id} in dev environment")
                    return cache_path

                # Verify checksum even for local files
                if not await self.verify_checksum(cache_path, dataset.checksum):
                    cache_path.unlink()
                    raise ValueError(f"Checksum verification failed for local dataset {dataset.id}")

                return cache_path
            else:
                # Create a dummy file for testing in local mode
                logger.warning(f"Local dataset file not found, creating dummy file for {dataset.id}")
                with open(cache_path, "w") as f:
                    # Create a simple JSONL file with test data
                    test_data = {
                        "id": f"test_{dataset.id}_1",
                        "question": f"Test question for {dataset.name}",
                        "answer": "Test answer",
                        "traits": dataset.traits,
                    }
                    f.write(json.dumps(test_data) + "\n")
                return cache_path

        # Cloud mode - download from URL
        headers = {}
        if auth_token and dataset.metadata.requires_auth:
            headers["Authorization"] = f"Bearer {auth_token}"

        dataset_url = f"{base_url}/{dataset.url}"

        async with aiohttp.ClientSession() as session, session.get(dataset_url, headers=headers) as response:
            response.raise_for_status()

            # Download in chunks
            with open(cache_path, "wb") as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)

        # Verify checksum
        if not await self.verify_checksum(cache_path, dataset.checksum):
            cache_path.unlink()
            raise ValueError(f"Checksum verification failed for {dataset.id}")

        return cache_path

    async def verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        """Verify file checksum.

        Args:
            file_path: Path to file
            expected_checksum: Expected SHA256 checksum

        Returns:
            True if checksum matches
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        actual_checksum = f"sha256:{sha256_hash.hexdigest()}"
        return actual_checksum == expected_checksum

    async def download_bundle(
        self, bundle_url: str, bundle_checksum: str, extract_path: Path, auth_token: Optional[str] = None
    ) -> Path:
        """Download and extract a bundle file.

        Args:
            bundle_url: URL of the bundle file
            bundle_checksum: Expected checksum of the bundle
            extract_path: Path to extract the bundle to
            auth_token: Optional authentication token

        Returns:
            Path to extracted bundle directory
        """
        bundle_path = extract_path.parent / f"{extract_path.name}.zip"
        bundle_path.parent.mkdir(parents=True, exist_ok=True)

        # Download bundle
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        async with aiohttp.ClientSession() as session, session.get(bundle_url, headers=headers) as response:
            response.raise_for_status()

            with open(bundle_path, "wb") as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)

        # Verify checksum
        if not await self.verify_checksum(bundle_path, bundle_checksum):
            bundle_path.unlink()
            raise ValueError("Bundle checksum verification failed")

        # Extract bundle
        extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)

        # Clean up bundle file
        bundle_path.unlink()

        return extract_path

    def import_dataset(self, dataset: Dataset, file_path: Path, db: Session):
        """Import a dataset into the database.

        Args:
            dataset: Dataset metadata
            file_path: Path to dataset file
            db: Database session
        """
        # TODO: Implement actual import logic
        # This would:
        # 1. Read the JSONL file
        # 2. Transform to database format
        # 3. Insert into exp_datasets table
        # 4. Update traits associations

        logger.info(f"Importing dataset {dataset.id} from {file_path}")

        # Placeholder for actual implementation
        with open(file_path, "r") as f:
            line_count = sum(1 for _ in f)
            logger.info(f"Dataset {dataset.id} contains {line_count} samples")

    async def sync_datasets(
        self, manifest: EvalDataManifest, current_version: str, force_sync: bool = False, use_bundles: bool = True
    ) -> Dict[str, Any]:
        """Sync datasets based on manifest.

        Args:
            manifest: Dataset manifest
            current_version: Current local version
            force_sync: Force sync even if versions match
            use_bundles: Whether to use bundle downloads when available

        Returns:
            Sync results
        """
        results = {
            "synced_datasets": [],
            "failed_datasets": [],
            "skipped_datasets": [],
            "total_size_mb": 0,
            "used_bundles": False,
        }

        # Check if sync is needed
        if not force_sync and manifest.version_info.current_version == current_version:
            logger.info("Versions match, no sync needed")
            return results

        # Get authentication token if needed
        auth_token = None
        if manifest.authentication:
            # TODO: Implement token fetching
            auth_token = "dummy_token"

        # Try to use complete bundle if available
        if use_bundles and manifest.repository.bundle_url and manifest.repository.bundle_checksum:
            try:
                logger.info("Downloading complete dataset bundle")
                bundle_dir = await self.download_bundle(
                    manifest.repository.bundle_url,
                    manifest.repository.bundle_checksum,
                    self.base_cache_dir / "bundles" / manifest.version_info.current_version,
                    auth_token,
                )
                results["used_bundles"] = True
                results["total_size_mb"] = manifest.repository.bundle_size_mb or 0

                # Import all datasets from bundle
                with Session(engine) as db:
                    for collection_name, collection in manifest.datasets.items():
                        for dataset in collection.datasets:
                            try:
                                # Find dataset file in bundle
                                dataset_file = bundle_dir / dataset.url
                                if dataset_file.exists():
                                    self.import_dataset(dataset, dataset_file, db)
                                    results["synced_datasets"].append(dataset.id)
                                else:
                                    logger.warning(f"Dataset {dataset.id} not found in bundle")
                            except Exception as e:
                                logger.error(f"Failed to import dataset {dataset.id} from bundle: {e}")
                                results["failed_datasets"].append({"dataset_id": dataset.id, "error": str(e)})

                    # Update version in database
                    # TODO: Update system_metadata with new version

                return results

            except Exception as e:
                logger.warning(f"Failed to use bundle download, falling back to individual downloads: {e}")
                results["used_bundles"] = False

        # Fall back to individual dataset downloads
        with Session(engine) as db:
            for collection_name, collection in manifest.datasets.items():
                logger.info(f"Processing {collection_name} datasets")

                # Try collection bundle first if available
                if use_bundles and collection.bundle_url and collection.bundle_checksum:
                    try:
                        logger.info(f"Downloading {collection_name} bundle")
                        bundle_dir = await self.download_bundle(
                            collection.bundle_url,
                            collection.bundle_checksum,
                            self.base_cache_dir / "bundles" / collection_name / collection.version,
                            auth_token if manifest.authentication and collection_name in manifest.authentication.required_for else None,
                        )

                        # Import datasets from collection bundle
                        for dataset in collection.datasets:
                            try:
                                dataset_file = bundle_dir / dataset.url
                                if dataset_file.exists():
                                    self.import_dataset(dataset, dataset_file, db)
                                    results["synced_datasets"].append(dataset.id)
                                    results["total_size_mb"] += dataset.size_mb
                                else:
                                    logger.warning(f"Dataset {dataset.id} not found in {collection_name} bundle")
                            except Exception as e:
                                logger.error(f"Failed to import dataset {dataset.id}: {e}")
                                results["failed_datasets"].append({"dataset_id": dataset.id, "error": str(e)})
                        continue
                    except Exception as e:
                        logger.warning(f"Failed to use {collection_name} bundle, downloading individually: {e}")

                # Download individual datasets
                for dataset in collection.datasets:
                    try:
                        # Check if dataset already exists
                        # TODO: Query database to check if dataset exists

                        logger.info(f"Downloading dataset {dataset.id}")
                        file_path = await self.download_dataset(dataset, manifest.repository.base_url, auth_token)

                        self.import_dataset(dataset, file_path, db)

                        results["synced_datasets"].append(dataset.id)
                        results["total_size_mb"] += dataset.size_mb

                    except Exception as e:
                        logger.error(f"Failed to sync dataset {dataset.id}: {e}")
                        results["failed_datasets"].append({"dataset_id": dataset.id, "error": str(e)})

            # Update version in database
            # TODO: Update system_metadata with new version

        return results

    async def run_migrations(self, manifest: EvalDataManifest, current_version: str):
        """Run database migrations if needed.

        Args:
            manifest: Dataset manifest
            current_version: Current version
        """
        if not manifest.migration or not manifest.requires_migration(current_version):
            logger.info("No migrations required")
            return

        logger.info(f"Running migrations from {manifest.migration.from_version} to {manifest.migration.to_version}")

        # TODO: Implement migration execution

    def record_sync_results(
        self, db: Session, manifest_version: str, sync_status: str, sync_metadata: Optional[Dict] = None
    ) -> None:
        """Record sync results in the database.

        Args:
            db: Database session
            manifest_version: Version that was synced
            sync_status: Status of the sync ('completed', 'failed', 'in_progress')
            sync_metadata: Additional metadata about the sync
        """
        from datetime import datetime

        from .models import EvalSyncState

        sync_state = EvalSyncState(
            manifest_version=manifest_version,
            sync_timestamp=datetime.utcnow().isoformat(),
            sync_status=sync_status,
            sync_metadata=sync_metadata or {},
        )

        db.add(sync_state)
        db.commit()

        logger.info(f"Recorded sync state: version={manifest_version}, status={sync_status}")
