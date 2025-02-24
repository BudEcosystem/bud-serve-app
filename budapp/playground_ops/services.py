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

"""The playground ops services. Contains business logic for playground ops."""

from typing import Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import status

from ..commons import logging
from ..commons.db_utils import SessionMixin
from ..commons.exceptions import ClientException
from ..credential_ops.crud import CredentialDataManager
from ..credential_ops.models import Credential as CredentialModel
from ..endpoint_ops.crud import EndpointDataManager
from ..endpoint_ops.models import Endpoint as EndpointModel
from ..project_ops.crud import ProjectDataManager


logger = logging.get_logger(__name__)


class PlaygroundService(SessionMixin):
    """Playground service."""

    async def get_all_playground_deployments(
        self,
        current_user_id: Optional[UUID] = None,
        api_key: Optional[str] = None,
        offset: int = 0,
        limit: int = 10,
        filters: Optional[Dict] = None,
        order_by: Optional[List] = None,
        search: bool = False,
    ) -> Tuple[List[EndpointModel], int]:
        """Get all playground deployments."""
        filters = filters or {}
        order_by = order_by or []

        project_ids = await self._get_authorized_project_ids(current_user_id, api_key)
        logger.debug("authorized project_ids: %s", project_ids)

        db_endpoints, count = await EndpointDataManager(self.session).get_all_playground_deployments(
            project_ids,
            offset,
            limit,
            filters,
            order_by,
            search,
        )
        logger.debug("found %s deployments", count)

        return db_endpoints, count

    async def _get_authorized_project_ids(
        self, current_user_id: Optional[UUID] = None, api_key: Optional[str] = None
    ) -> List[UUID]:
        """Get all authorized project ids."""
        if current_user_id:
            # NOTE: As per user permissions list the playground deployments (accessible project ids)
            # TODO: Query all accessible project ids for the user (Currently all active project ids since permissions are not implemented)
            logger.debug(f"Getting all playground deployments for user {current_user_id}")
            return await ProjectDataManager(self.session).get_all_active_project_ids()
        elif api_key:
            # if api_key is present identify the project id
            db_credential = await CredentialDataManager(self.session).retrieve_by_fields(
                CredentialModel, fields={"key": api_key}, missing_ok=True
            )

            if not db_credential:
                logger.error("Invalid API key found")
                raise ClientException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    message="Invalid API key",
                )
            else:
                return [db_credential.project.id]
        else:
            raise ClientException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                message="Unauthorized to access this resource",
            )
