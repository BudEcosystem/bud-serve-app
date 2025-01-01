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
from ..commons.constants import ProjectStatusEnum, UserStatusEnum
from ..user_ops.models import User

logger = logging.get_logger(__name__)


class ProjectDataManager(DataManagerUtils):
    """Data manager for the Project model."""

    def get_unique_user_count_in_all_projects(self) -> int:
        """
        Get the count of unique users across all active projects.

        Returns:
            int: Count of unique users in all active projects.
        """
        unique_users_stmt = (
            select(func.count(distinct(project_user_association.c.user_id)))
            .join(Project, project_user_association.c.project_id == Project.id)
            .join(User, project_user_association.c.user_id == User.id)
            .where(
                Project.status == ProjectStatusEnum.ACTIVE,
                User.status != UserStatusEnum.DELETED,
            )
        )
        return self.scalar_one_or_none(unique_users_stmt) or 0
