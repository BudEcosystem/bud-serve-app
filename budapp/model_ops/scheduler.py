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

from ..commons.config import app_settings
import aiohttp
from budmicroframe.commons import logging
from typing import List, Dict

logger = logging.get_logger(__name__)


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
