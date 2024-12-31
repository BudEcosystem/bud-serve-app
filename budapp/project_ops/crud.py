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

"""The crud package, containing essential business logic, services, and routing configurations for the project ops."""

from uuid import UUID
from typing import Tuple
from sqlalchemy import func, distinct, select

from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils
from .models import project_user_association, Project
from ..commons.constants import ProjectStatusEnum

logger = logging.get_logger(__name__)


class ProjectDataManager(DataManagerUtils):
    """Data manager for the Project model."""

    async def get_project_count_by_user(self, user_id: UUID) -> Tuple[list[UUID], int]:
        """
        Get the list of active project IDs a user is part of.
        Args:
            user_id (UUID): The ID of the user.
        Returns:
            list[UUID]: A list of active project IDs the user is part of.
        """
        project_ids_stmt = (
            select(distinct(project_user_association.c.project_id))
            .join(Project, project_user_association.c.project_id == Project.id)
            .where(project_user_association.c.user_id == user_id, Project.status == ProjectStatusEnum.ACTIVE)
        )
        db_project_ids = [row for row in self.scalars_all(project_ids_stmt)]
        return db_project_ids, len(db_project_ids)

    async def get_unique_user_count_in_projects(self, project_ids: list[UUID]) -> int:
        """
        Get the count of unique users in the specified active projects.
        Args:
            project_ids (list[UUID]): A list of project IDs.
        Returns:
            int: Count of unique users in those active projects.
        """
        if not project_ids:
            return 0

        unique_users_stmt = (
            select(func.count(distinct(project_user_association.c.user_id)))
            .join(Project, project_user_association.c.project_id == Project.id)
            .where(project_user_association.c.project_id.in_(project_ids), Project.status == ProjectStatusEnum.ACTIVE)
        )
        return self.scalar_one_or_none(unique_users_stmt) or 0
