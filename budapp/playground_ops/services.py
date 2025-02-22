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
        if not filters:
            filters = {}
        if not order_by:
            order_by = []

        project_ids = []
        if current_user_id:
            # TODO: As per user permissions list the playground deployments (accessible project ids)
            logger.debug(f"Getting all playground deployments for user {current_user_id}")
            # TODO: Query all accessible project ids for the user
            db_project_ids = await ProjectDataManager(self.session).get_all_active_project_ids()
            project_ids = db_project_ids
        elif api_key:
            # if api_key is present identify the project id
            db_credential = await CredentialDataManager(self.session).retrieve_by_fields(
                CredentialModel, fields={"key": api_key}, missing_ok=True
            )

            if not db_credential:
                logger.error("Invalid API key found")
                raise ClientException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    message="Invalid API key",
                )
            else:
                project_id = db_credential.project.id
                project_ids = [project_id]
                logger.debug(f"Getting all playground deployments for project {project_id} based on API key")
        else:
            raise ClientException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                message="Unauthorized to access this resource",
            )

        db_endpoints, count = await EndpointDataManager(self.session).get_all_playground_deployments(
            project_ids,
            offset,
            limit,
            filters,
            order_by,
            search,
        )
        logger.debug("found %s deployments", len(db_endpoints))

        return db_endpoints, count
