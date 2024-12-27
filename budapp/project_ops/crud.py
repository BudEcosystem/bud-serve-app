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
from sqlalchemy import func, distinct, select

from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils
from .models import project_user_association

logger = logging.get_logger(__name__)


class ProjectDataManager(DataManagerUtils):
    """Data manager for the Project model."""

    async def get_user_project_stats(self, user_id: UUID) -> tuple[int, int]:
        """
        Get the count of total projects a user is part of and the number of unique users
        in those projects.

        Args:
            user_id (UUID): The ID of the user.

        Returns:
            tuple[int, int]: A tuple containing:
                - Count of total projects the user is present in.
                - Count of unique users present in those projects.
        """

        # Query to count the total projects the user is part of
        projects_count_stmt = select(func.count(distinct(project_user_association.c.project_id))).where(
            project_user_association.c.user_id == user_id
        )
        projects_count = self.scalar_one_or_none(projects_count_stmt) or 0

        # Query to count unique users in the projects the user is part of
        users_count_stmt = select(func.count(distinct(project_user_association.c.user_id))).where(
            project_user_association.c.project_id.in_(
                select(project_user_association.c.project_id).where(project_user_association.c.user_id == user_id)
            )
        )
        users_count = self.scalar_one_or_none(users_count_stmt) or 0

        return projects_count, users_count
