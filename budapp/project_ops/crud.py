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

from typing import Dict, List, Optional
from uuid import UUID

from fastapi import status
from fastapi.exceptions import HTTPException
from sqlalchemy import String as SqlAlchemyString
from sqlalchemy import cast, distinct, func, select

from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils

from ..commons.constants import ProjectStatusEnum, UserStatusEnum
from ..user_ops.models import User
from .models import Project, project_user_association


logger = logging.get_logger(__name__)


class ProjectDataManager(DataManagerUtils):
    """Data manager for the Project model."""

    def get_unique_user_count_in_all_projects(self) -> int:
        """Get the count of unique users across all active projects.

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

    async def get_all_active_project_ids(self) -> List[UUID]:
        """Get all active project ids.

        Returns:
            List[UUID]: List of active project ids.
        """
        stmt = select(Project.id).where(Project.status == ProjectStatusEnum.ACTIVE)
        return self.scalars_all(stmt)

    async def retrieve_project_by_fields(
        self,
        fields: Dict,
        missing_ok: bool = False,
        case_sensitive: bool = True,
    ) -> Optional[Project]:
        """Retrieve project by fields."""
        await self.validate_fields(Project, fields)

        if case_sensitive:
            stmt = select(Project).filter_by(**fields)
        else:
            conditions = []
            for field_name, value in fields.items():
                field = getattr(Project, field_name)
                if isinstance(field.type, SqlAlchemyString):
                    conditions.append(func.lower(cast(field, SqlAlchemyString)) == func.lower(value))
                else:
                    conditions.append(field == value)
            stmt = select(Project).filter(*conditions)

        db_project = await self.scalar_one_or_none(stmt)

        if not missing_ok and db_project is None:
            logger.info("Project not found in database")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

        return db_project if db_project else None

    async def get_active_user_ids_in_project(self, project_id: UUID) -> List[User]:
        """Get all active users in a project."""
        stmt = (
            select(User.id)
            .select_from(Project)
            .filter_by(id=project_id)
            .outerjoin(
                project_user_association,
                Project.id == project_user_association.c.project_id,
            )
            .outerjoin(User, project_user_association.c.user_id == User.id)
            .filter_by(status=UserStatusEnum.ACTIVE)
        )

        return await self.scalars_all(stmt)
