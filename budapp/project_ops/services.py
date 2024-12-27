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

"""The project ops services. Contains business logic for project ops."""

from typing import Any, Dict, Tuple
from uuid import UUID

from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from ..commons.constants import ProjectStatusEnum

from .crud import ProjectDataManager
from .models import Project as ProjectModel
from .schemas import ProjectResponse

logger = logging.get_logger(__name__)


class ProjectService(SessionMixin):
    """Project service."""

    async def edit_project(self, project_id: UUID, data: Dict[str, Any]) -> ProjectResponse:
        """Edit project by validating and updating specific fields."""
        # Retrieve existing model
        db_project = await ProjectDataManager(self.session).retrieve_by_fields(
            model=ProjectModel,
            fields={"id": project_id, "status": ProjectStatusEnum.ACTIVE},
        )

        if "name" in data:
            duplicate_project = await ProjectDataManager(self.session).retrieve_by_fields(
                model=ProjectModel,
                fields={"name": data["name"], "status": ProjectStatusEnum.ACTIVE},
                exclude_fields={"id": project_id},
                missing_ok=True,
            )
            if duplicate_project:
                raise ClientException("Project name already exists")

        db_project = await ProjectDataManager(self.session).update_by_fields(db_project, data)

        return db_project

    async def get_user_project_stats(self, user_id: UUID) -> Tuple[int, int]:
        """
        Retrieve the counts of projects user is present in and total members.
        Returns:
            tuple[int, int]: A tuple containing:
                - Count of projects user is present in
                - Count of users in the projects
        """

        return await ProjectDataManager(self.session).get_user_project_stats(user_id)
