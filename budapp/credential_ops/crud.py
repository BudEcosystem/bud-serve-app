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

"""The crud package, containing essential business logic, services, and routing configurations for the credential ops."""

from uuid import UUID
from sqlalchemy import select

from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils
from budapp.credential_ops.models import CloudProviders
from budapp.credential_ops.models import CloudCredentials
from typing import Optional
# from budapp.credential_ops.schemas import CloudProvidersCreateRequest


logger = logging.get_logger(__name__)


class CredentialDataManager(DataManagerUtils):
    """Data manager for the Credential model."""

    pass


class ProprietaryCredentialDataManager(DataManagerUtils):
    """Data manager for the ProprietaryCredential model."""

    pass

class CloudProviderDataManager(DataManagerUtils):
    """Data manager for the CloudProvider model."""

    async def get_all_providers(self) -> list[CloudProviders]:
        """Get all cloud providers."""
        stmt = select(CloudProviders)
        return self.scalars_all(stmt)

class CloudProviderCredentialDataManager(DataManagerUtils):
    """Data manager for the CloudProviderCredential model."""

    async def get_credentials_by_user(self, user_id: UUID, provider_id: Optional[UUID] = None) -> list[CloudCredentials]:
        """
        Get cloud provider credentials by user ID, optionally filtered by provider ID.

        Args:
            user_id: The ID of the user whose credentials to retrieve
            provider_id: Optional provider ID to filter credentials by

        Returns:
            List of CloudCredentials that match the criteria
        """
        stmt = select(CloudCredentials).where(CloudCredentials.user_id == user_id)

        # If provider_id is provided, add it to the query filter
        if provider_id:
            stmt = stmt.where(CloudCredentials.provider_id == provider_id)

        result = self.scalars_all(stmt)
        return result
