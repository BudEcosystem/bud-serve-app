"""Service for syncing evaluation datasets from cloud repository."""

import hashlib
import zipfile
from pathlib import Path
from typing import Dict, Optional

import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession

from ..commons import logging
from ..commons.database import SessionLocal
from .manifest_schemas import Dataset, DatasetCollection, EvalDataManifest


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
        """Fetch and parse the manifest from the cloud repository.

        Args:
            manifest_url: URL of the manifest file

        Returns:
            Parsed manifest object
        """
        async with aiohttp.ClientSession() as session, session.get(manifest_url) as response:
            response.raise_for_status()
            manifest_data = await response.json()
            return EvalDataManifest(**manifest_data)

    async def get_current_version(self, db: AsyncSession) -> Optional[str]:
        """Get the current version from the database.

        Args:
            db: Database session

        Returns:
            Current version string or None if not set
        """
        # TODO: Implement actual database query
        # This would query a system_metadata table or similar
        return "2024.01.01"  # Placeholder

    async def download_dataset(self, dataset: Dataset, base_url: str, auth_token: Optional[str] = None) -> Path:
        """Download a dataset file.

        Args:
            dataset: Dataset information
            base_url: Base URL for downloads
            auth_token: Optional authentication token

        Returns:
            Path to downloaded file
        """
        headers = {}
        if auth_token and dataset.metadata.requires_auth:
            headers["Authorization"] = f"Bearer {auth_token}"

        dataset_url = f"{base_url}/{dataset.url}"
        cache_path = self.base_cache_dir / dataset.url
        cache_path.parent.mkdir(parents=True, exist_ok=True)

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
        self, bundle_url: str, checksum: str, target_dir: Path, auth_token: Optional[str] = None
    ) -> Path:
        """Download and extract a bundle zip file.

        Args:
            bundle_url: URL of the bundle zip file
            checksum: Expected checksum of the bundle
            target_dir: Directory to extract bundle contents
            auth_token: Optional authentication token

        Returns:
            Path to the extracted bundle directory
        """
        headers = {}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        # Download bundle to temp location
        bundle_path = self.base_cache_dir / "temp" / Path(bundle_url).name
        bundle_path.parent.mkdir(parents=True, exist_ok=True)

        async with aiohttp.ClientSession() as session, session.get(bundle_url, headers=headers) as response:
            response.raise_for_status()

            # Download in chunks
            with open(bundle_path, "wb") as f:
                async for chunk in response.content.iter_chunked(8192):
                    f.write(chunk)

        # Verify checksum
        if not await self.verify_checksum(bundle_path, checksum):
            bundle_path.unlink()
            raise ValueError(f"Checksum verification failed for bundle {bundle_url}")

        # Extract bundle
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bundle_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)

        # Clean up temp file
        bundle_path.unlink()

        return target_dir

    async def import_dataset(self, dataset: Dataset, file_path: Path, db: AsyncSession):
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
    ) -> Dict[str, any]:
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
        if use_bundles and manifest.repository.bundle_url:
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
                async with SessionLocal() as db:
                    for collection_name, collection in manifest.datasets.items():
                        for dataset in collection.datasets:
                            try:
                                # Find dataset file in bundle
                                dataset_file = bundle_dir / dataset.url
                                if dataset_file.exists():
                                    await self.import_dataset(dataset, dataset_file, db)
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
        async with SessionLocal() as db:
            for collection_name, collection in manifest.datasets.items():
                logger.info(f"Processing {collection_name} datasets")

                # Try collection bundle first if available
                if use_bundles and collection.bundle_url:
                    try:
                        logger.info(f"Downloading {collection_name} bundle")
                        bundle_dir = await self.download_bundle(
                            collection.bundle_url,
                            collection.bundle_checksum,
                            self.base_cache_dir / "bundles" / collection_name / collection.version,
                            auth_token if collection_name in manifest.authentication.required_for else None,
                        )

                        # Import datasets from collection bundle
                        for dataset in collection.datasets:
                            try:
                                dataset_file = bundle_dir / dataset.url
                                if dataset_file.exists():
                                    await self.import_dataset(dataset, dataset_file, db)
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

                        await self.import_dataset(dataset, file_path, db)

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
        # This would download and execute SQL scripts in order
        for script in manifest.migration.scripts:
            logger.info(f"Executing migration: {script.description}")
            # Download and execute script
