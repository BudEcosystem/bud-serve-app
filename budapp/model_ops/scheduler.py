#  -----------------------------------------------------------------------------
#  Copyright (c) 2024 Bud Ecosystem Inc.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  -----------------------------------------------------------------------------

"""The model ops scheduler. Contains business logic for model ops."""

import json
from typing import Dict, List

import aiohttp
from budmicroframe.commons import logging
from sqlalchemy.orm import Session

from ..commons.config import app_settings
from ..commons.database import engine
from ..endpoint_ops.crud import EndpointDataManager
from ..model_ops.crud import CloudModelDataManager, ModelDataManager, ProviderDataManager


logger = logging.get_logger(__name__)
from budapp.initializers.provider_seeder import PROVIDERS_SEEDER_FILE_PATH

from ..model_ops.models import CloudModel as CloudModelModel
from ..model_ops.models import Provider as ProviderModel
from ..model_ops.schemas import CloudModelCreate, ProviderCreate


class CloudModelSyncScheduler:
    """Schedule cloud model db with cloud service."""

    @staticmethod
    async def get_latest_compatible_models() -> List[Dict]:
        """Get the latest compatible models from the cloud service.

        Returns:
            List[Dict]: List of compatible models.
        """
        PAGE_LIMIT = 5
        api_endpoint = f"{app_settings.bud_connect_base_url}model/get-compatible-models"
        params = {
            "limit": PAGE_LIMIT,
            "engine": app_settings.cloud_model_seeder_engine,
        }

        try:
            data = []

            async with aiohttp.ClientSession() as session:
                # First request to get total pages
                params["page"] = 1
                async with session.get(api_endpoint, params=params) as response:
                    response_data = await response.json()
                    total_pages = response_data.get("total_pages", 0)
                    logger.debug("Total pages: %s", total_pages)
                    data.extend(response_data.get("items", []))

                # Fetch remaining pages
                for page in range(2, total_pages + 1):
                    params["page"] = page
                    async with session.get(api_endpoint, params=params) as response:
                        response_data = await response.json()
                        cloud_providers = response_data.get("items", [])
                        logger.debug("Found %s providers on page %s", len(cloud_providers), page)
                        data.extend(cloud_providers)

            return data
        except Exception as e:
            logger.error("Error getting latest compatible models: %s", e)
            return []

    async def sync_data():
        """Sync the data from the cloud service."""
        providers = await CloudModelSyncScheduler.get_latest_compatible_models()
        logger.debug("Found %s providers from cloud service", len(providers))

        # Set is_active to False for previous version of providers
        provider_types = [provider["provider_type"] for provider in providers]
        logger.debug("Provider types: %s", provider_types)

        # Update is_active to False for previous version of providers
        with Session(engine) as session:
            await ProviderDataManager(session).soft_delete_non_supported_providers(provider_types)
            logger.debug("Soft deleted non-supported providers")

        # Upsert new providers
        provider_type_id_mapper = {}
        with Session(engine) as session:
            for provider in providers:
                provider_data = ProviderCreate(
                    name=provider["name"],
                    description=provider["description"],
                    type=provider["provider_type"],
                    icon=provider["icon"],
                ).model_dump()
                db_provider = await ProviderDataManager(session).upsert_one(ProviderModel, provider_data, ["type"])
                provider_type_id_mapper[provider["provider_type"]] = str(db_provider.id)
                logger.debug("Upserted provider: %s", db_provider.id)

        # Save to json file by following earlier implementation
        # NOTE: this json is used in proprietary/credentials/provider-info api
        # TODO: Move this implementation to db
        providers_data = {}
        for provider in providers:
            providers_data[provider["provider_type"]] = {
                "name": provider["name"],
                "type": provider["provider_type"],
                "description": provider["description"],
                "icon": provider["icon"],
                "credentials": provider["credentials"],
            }
        with open(PROVIDERS_SEEDER_FILE_PATH, "w") as f:
            json.dump(providers_data, f, indent=4)
        logger.debug("Saved providers to %s", PROVIDERS_SEEDER_FILE_PATH)

        # Get all cloud model uris
        cloud_model_uris = []
        cloud_model_data = []
        for provider in providers:
            cloud_models = provider["models"]
            for cloud_model in cloud_models:
                cloud_model_uris.append(cloud_model["uri"])
                max_input_tokens = (
                    cloud_model["tokens"].get("max_input_tokens", None) if cloud_model["tokens"] is not None else None
                )
                cloud_model_data.append(
                    CloudModelCreate(
                        provider_id=provider_type_id_mapper[provider["provider_type"]],
                        uri=cloud_model["uri"],
                        name=cloud_model["uri"],
                        modality="llm",  # TODO: change to cloud_model["modality"]
                        source=provider["provider_type"],
                        max_input_tokens=max_input_tokens,
                        input_cost=cloud_model["input_cost"],
                        output_cost=cloud_model["output_cost"],
                        supported_endpoints=cloud_model["endpoints"],
                        deprecation_date=cloud_model["deprecation_date"],
                    )
                )

        # Get model ids from model zoo of deprecated cloud models
        deprecated_model_ids = []
        with Session(engine) as session:
            deprecated_models = await ModelDataManager(session).get_deprecated_cloud_models(cloud_model_uris)
            deprecated_model_ids = [db_model.id for db_model in deprecated_models]
            logger.debug("Found %s deprecated cloud models", deprecated_model_ids)

        # Mark endpoints as deprecated
        with Session(engine) as session:
            await EndpointDataManager(session).mark_as_deprecated(deprecated_model_ids)
            logger.debug("Marked endpoints as deprecated")

        # Soft delete deprecated models
        with Session(engine) as session:
            await ModelDataManager(session).soft_delete_deprecated_models(deprecated_model_ids)
            logger.debug("Soft deleted deprecated models from model zoo")

        # Remove deprecated cloud models
        with Session(engine) as session:
            await CloudModelDataManager(session).remove_non_supported_cloud_models(cloud_model_uris)
            logger.debug("Removed non-supported cloud models")

        # Upsert new cloud models
        with Session(engine) as session:
            for cloud_model in cloud_model_data:
                await CloudModelDataManager(session).upsert_one(CloudModelModel, cloud_model.model_dump(), ["uri"])
                logger.debug("Upserted cloud model: %s", cloud_model)
        logger.debug("Upserted %s cloud models", len(cloud_model_data))


if __name__ == "__main__":
    import asyncio

    asyncio.run(CloudModelSyncScheduler.sync_data())

    # python -m budapp.model_ops.scheduler
