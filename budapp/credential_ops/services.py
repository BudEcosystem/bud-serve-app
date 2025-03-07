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

from budapp.cluster_ops.utils import logging
from budapp.commons.db_utils import SessionMixin
from budapp.credential_ops.crud import CloudProviderDataManager
from budapp.credential_ops.models import CloudCredentials, CloudProviders
from budapp.credential_ops.schemas import CloudProvidersCreateRequest
import json


logger = logging.get_logger(__name__)


class ClusterProviderService(SessionMixin):
    """ClusterProviderService is a service class that provides cluster-related operations."""

    async def create_provider_credential(self, req: CloudProvidersCreateRequest) -> None:
        """Create a new credential for a provider."""
        try:
            # Get the provider from the database
            provider = await CloudProviderDataManager(self.session).retrieve_by_fields(
                CloudProviders, {"id": req.provider_id}
            )

            # Validate the provider
            if not provider:
                raise ValueError(f"Provider with id {req.provider_id} not found")

            # Parse the schema definition from the provider
            schema = json.loads(provider.schema_definition) if provider.schema_definition else {}

            # Get the required fields from the schema
            required_fields = schema.get("required", [])

            # Validate the required fields
            for field in required_fields:
                if field not in req.credential_values:
                    raise ValueError(f"Required field '{field}' is missing in the credential values")

            # Save the credential values
            cloud_credential = CloudCredentials(
                provider_id=req.provider_id,
                credential_values=req.credential_values  # Assuming CloudCredentials has a field for credential_values
            )
            await CloudProviderDataManager(self.session).insert_one(cloud_credential)

            logger.debug(f"Created credential for provider {cloud_credential.id}")
        except Exception as e:
            logger.error(f"Failed to create credential for provider {req.provider_id}: {e}")
            raise e
