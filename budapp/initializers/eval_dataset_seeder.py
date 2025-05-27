import json
import os
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from budapp.commons import logging
from budapp.commons.database import engine
from budapp.eval_ops.models import ExpDataset as DatasetModel
from budapp.eval_ops.models import ExpTrait as TraitModel
from budapp.eval_ops.models import ExpTraitsDatasetPivot as PivotModel

from .base_seeder import BaseSeeder


logger = logging.get_logger(__name__)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_SEEDER_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "data", "eval_dataset_seeder.json")

class EvalDatasetSeeder(BaseSeeder):
    """Seeder for the ExpDataset model."""

    async def seed(self) -> None:
        """Seed evaluation datasets to the database."""
        with Session(engine) as session:
            try:
                # Note: We're not awaiting here, just calling the method
                self._seed_datasets(session)
            except Exception as e:
                logger.exception(f"Failed to seed datasets: {e}")

    def _seed_datasets(self, session: Session) -> None:
        """Seed datasets to the database."""
        datasets_data = self._get_datasets_data()
        logger.debug(f"Found {len(datasets_data)} datasets in the seeder file")

        # Get all traits and create case-insensitive lookup
        all_traits = session.query(TraitModel).all()
        # Create two lookups - exact match and lowercase for case-insensitive matching
        traits_by_name = {trait.name: trait for trait in all_traits}
        traits_by_name_lower = {trait.name.lower(): trait for trait in all_traits}
        logger.debug(f"Found {len(all_traits)} traits in the database")

        # Get existing datasets by name for efficient lookup
        existing_datasets = {dataset.name: dataset for dataset in session.query(DatasetModel).all()}
        logger.debug(f"Found {len(existing_datasets)} datasets in the database")

        # Get existing pivot entries to avoid duplicates
        existing_pivots = set()
        for pivot in session.query(PivotModel).all():
            existing_pivots.add((str(pivot.trait_id), str(pivot.dataset_id)))

        updated = 0
        created = 0
        skipped = 0
        pivots_created = 0

        for dataset in datasets_data:
            name = dataset["name"]

            # Extract dimensions (traits) from the dataset
            trait_names = []
            if "dimensions" in dataset and dataset["dimensions"]:
                for dimension in dataset["dimensions"]:
                    if "en" in dimension:
                        trait_names.append(dimension["en"])

            # Skip if no valid traits
            if not trait_names:
                logger.warning(f"Dataset {name} has no traits specified, skipping")
                skipped += 1
                continue

            # Prepare dataset fields
            dataset_fields = {
                "name": name,
                "description": dataset.get("desc", {}).get("en") if dataset.get("desc") else None,
                "meta_links": {
                    "github": dataset.get("githubLink", None),
                    "paper": dataset.get("paperLink", None),
                    "website": dataset.get("websiteLink", None),
                },
                "estimated_input_tokens": dataset.get("estimated_input_tokens"),
                "estimated_output_tokens": dataset.get("estimated_output_tokens"),
                "language": dataset.get("language"),
                "domains": dataset.get("domains"),
                "concepts": dataset.get("concepts"),
                "humans_vs_llm_qualifications": dataset.get("humans_vs_llm_qualifications"),
                "task_type": dataset.get("task_type"),
                "modalities": dataset.get("modalities"),
            }

            # Add config_validation_schema if needed
            if "config_validation_schema" in dataset:
                dataset_fields["config_validation_schema"] = dataset["config_validation_schema"]

            # Update or create dataset
            if name in existing_datasets:
                db_dataset = existing_datasets[name]

                # Update fields
                for key, value in dataset_fields.items():
                    if value is not None:
                        setattr(db_dataset, key, value)

                updated += 1
                logger.info(f"Updated dataset {name}")
            else:
                # Create new dataset
                db_dataset = DatasetModel(**dataset_fields)
                session.add(db_dataset)
                session.flush()  # To get the dataset ID
                created += 1
                logger.info(f"Created new dataset {name}")

            # Now handle the traits through the pivot table
            for trait_name in trait_names:
                # Try exact match first, then case-insensitive match
                trait = None
                if trait_name in traits_by_name:
                    trait = traits_by_name[trait_name]
                elif trait_name.lower() in traits_by_name_lower:
                    trait = traits_by_name_lower[trait_name.lower()]
                    logger.info(f"Found case-insensitive match for trait '{trait_name}' -> '{trait.name}'")

                if trait is None:
                    logger.warning(f"Trait '{trait_name}' not found in database (case-insensitive search), skipping for dataset {name}")
                    continue

                trait_id = trait.id

                # Check if this pivot already exists
                pivot_key = (str(trait_id), str(db_dataset.id))
                if pivot_key not in existing_pivots:
                    # Create new pivot entry
                    pivot = PivotModel(trait_id=trait_id, dataset_id=db_dataset.id)
                    session.add(pivot)
                    existing_pivots.add(pivot_key)
                    pivots_created += 1
                    logger.info(f"Created new pivot for dataset {name} and trait {trait.name}")

        # Commit all changes at once
        session.commit()
        logger.debug(f"Updated {updated} datasets, created {created} new datasets, skipped {skipped} datasets, created {pivots_created} trait-dataset pivots")

    def _get_datasets_data(self) -> List[Dict[str, Any]]:
        """Get datasets data from the seeder file."""
        try:
            with open(DATASET_SEEDER_FILE_PATH, "r") as file:
                return json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {DATASET_SEEDER_FILE_PATH}") from e
