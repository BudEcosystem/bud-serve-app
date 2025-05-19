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
from ..model_ops.crud import ProviderDataManager


logger = logging.get_logger(__name__)
from budapp.initializers.provider_seeder import PROVIDERS_SEEDER_FILE_PATH

from ..model_ops.models import Provider as ProviderModel
from ..model_ops.schemas import ProviderCreate


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
        with Session(engine) as session:
            for provider in providers:
                provider_data = ProviderCreate(
                    name=provider["name"],
                    description=provider["description"],
                    type=provider["provider_type"],
                    icon=provider["icon"],
                ).model_dump()
                await ProviderDataManager(session).upsert_one(ProviderModel, provider_data, ["type"])
                logger.debug("Upserted provider: %s", provider["provider_type"])

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

        # Collect all cloud models uri
        new_cloud_model_uris = []
        for provider in providers:
            for model in provider["models"]:
                new_cloud_model_uris.append(model["uri"])
        logger.debug("New cloud model uris: %s", new_cloud_model_uris)


if __name__ == "__main__":
    import asyncio

    asyncio.run(CloudModelSyncScheduler.sync_data())

    # python -m budapp.model_ops.scheduler
