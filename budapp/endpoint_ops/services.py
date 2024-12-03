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

"""The endpoint ops services. Contains business logic for endpoint ops."""

from typing import Dict, List, Tuple
from uuid import UUID

from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin
from budapp.project_ops.crud import ProjectDataManager
from budapp.project_ops.models import Project as ProjectModel

from .crud import EndpointDataManager
from .models import Endpoint as EndpointModel


logger = logging.get_logger(__name__)


class EndpointService(SessionMixin):
    """Endpoint service."""

    async def get_all_endpoints(
        self,
        project_id: UUID,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[EndpointModel], int]:
        """Get all active endpoints."""
        if search:
            # Only include name, exclude other filters
            # Otherwise it will perform global search on all fields
            filters.pop("status", None)

        # Validate project_id
        await ProjectDataManager(self.session).retrieve_by_fields(ProjectModel, {"id": project_id})

        return await EndpointDataManager(self.session).get_all_active_endpoints(
            project_id, offset, limit, filters, order_by, search
        )
