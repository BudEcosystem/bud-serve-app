import json
import os
from typing import Any, Dict

from sqlalchemy.orm import Session

from budapp.commons import logging
from budapp.commons.config import app_settings
from budapp.commons.constants import ModelProviderTypeEnum
from budapp.commons.database import engine
from budapp.model_ops.crud import ModelDataManager
from budapp.model_ops.models import Model
from budapp.user_ops.crud import UserDataManager
from budapp.user_ops.models import User as UserModel

from .base_seeder import BaseSeeder


logger = logging.get_logger(__name__)

# current file path
CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))

# seeder file path
CLOUD_MODEL_SEEDER_FILE_PATH = os.path.join(CURRENT_FILE_PATH, "data", "cloud_model_seeder.json")


class CloudModelSeeder(BaseSeeder):
    """Cloud model seeder."""

    async def seed(self):
        """Seed the cloud models."""
        with Session(engine) as session:
            try:
                await self._seed_cloud_models(session)
            except Exception as e:
                logger.exception(f"Failed to seed cloud models: {e}")

    @staticmethod
    async def _seed_cloud_models(session: Session) -> None:
        """Seed the cloud models."""
        db_user = await UserDataManager(session).retrieve_by_fields(
            UserModel,
            {"email": app_settings.superuser_email, "is_active": True, "is_superuser": True},
            missing_ok=True,
        )

        if db_user:
            cloud_model_data = await CloudModelSeeder._get_cloud_model_data()
            logger.debug(f"Seeding cloud models for {len(cloud_model_data)} providers")

            for provider, model_data in cloud_model_data.items():
                # URI as key and details as value
                provider_model = {}
                for model in model_data["models"]:
                    provider_model[model["uri"]] = model

                # Check if the models already exist
                existing_models = await ModelDataManager(session).get_all_models_by_source_uris(
                    provider, list(provider_model.keys())
                )
                logger.debug(f"Found {len(existing_models)} existing models for provider {provider}. Updating...")

                # Update the existing models
                for existing_model in existing_models:
                    update_data = {
                        "name": provider_model[existing_model.uri]["name"],
                        "modality": provider_model[existing_model.uri]["modality"],
                        "type": provider_model[existing_model.uri]["type"],
                        "source": provider_model[existing_model.uri]["source"],
                        "uri": provider_model[existing_model.uri]["uri"],
                        "icon": provider_model[existing_model.uri]["icon"],
                        "provider_type": ModelProviderTypeEnum.CLOUD_MODEL.value,
                        "created_by": db_user.id,
                    }

                    # Update existing model in the database
                    await ModelDataManager(session).update_by_fields(existing_model, update_data)

                    # Remove the model from the provider_model
                    del provider_model[existing_model.uri]

                # Bulk insert the new models
                new_models = [
                    Model(**model, provider_type=ModelProviderTypeEnum.CLOUD_MODEL.value, created_by=db_user.id)
                    for model in provider_model.values()
                ]
                await ModelDataManager(session).insert_all(new_models)
                logger.debug(f"Seeded {len(new_models)} new models for provider {provider}")
        else:
            logger.error("Super user not found. Skipping cloud model seeding.")

    @staticmethod
    async def _get_cloud_model_data() -> Dict[str, Any]:
        """Get cloud_model data from the database."""
        try:
            with open(CLOUD_MODEL_SEEDER_FILE_PATH, "r") as file:
                return json.load(file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"File not found: {CLOUD_MODEL_SEEDER_FILE_PATH}") from e
