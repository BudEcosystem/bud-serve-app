import json
import os
from typing import Any, List

from sqlalchemy.orm import Session

from budapp.commons import logging
from budapp.commons.database import engine
from budapp.eval_ops.models import ExpTrait as TraitModel

from .base_seeder import BaseSeeder


logger = logging.get_logger(__name__)

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
TRAITS_SEEDER_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "data", "traits_seeder.json")

class TraitsSeeder(BaseSeeder):
    """Seeder for the Trait model."""

    async def seed(self) -> None:
        """Seed traits to the database."""
        with Session(engine) as session:
            try:
                await self._seed_traits(session)
            except Exception as e:
                logger.exception(f"Failed to seed traits: {e}")

    @staticmethod
    async def _seed_traits(session: Session) -> None:
        """Seed traits to the database."""
        traits_data = await TraitsSeeder._async_get_traits_data()
        logger.debug(f"Found {len(traits_data)} traits in the seeder file")

        # Get all existing traits by name
        existing_traits = {t.name: t for t in session.query(TraitModel).all()}
        logger.debug(f"Found {len(existing_traits)} traits in the database")

        updated = 0
        created = 0
        for trait in traits_data:
            name = trait["name"]
            if name in existing_traits:
                db_trait = existing_traits[name]
                db_trait.description = trait.get("description", db_trait.description)
                db_trait.icon = trait.get("icon", db_trait.icon)
                updated += 1
                logger.info(f"Updated trait {name} with id {db_trait.id}")
            else:
                new_trait = TraitModel(
                    name=trait["name"],
                    description=trait.get("description"),
                    icon=trait.get("icon"),
                )
                session.add(new_trait)
                created += 1
                logger.info(f"Created new trait {name}")
        session.commit()
        logger.debug(f"Updated {updated} traits, created {created} new traits")

    @staticmethod
    async def _async_get_traits_data() -> List[Any]:
        """Get traits data from the seeder file."""
        try:
            with open(TRAITS_SEEDER_FILE_PATH, "r") as file:
                return json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {TRAITS_SEEDER_FILE_PATH}") from e
