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

"""The core package, containing essential business logic, services, and routing configurations for the permissions."""

from typing import List, Union
from uuid import UUID
from sqlalchemy import delete
from budapp.commons import logging
from budapp.commons.db_utils import DataManagerUtils

from .models import ProjectPermission as ProjectPermissionModel

logger = logging.get_logger(__name__)


class PermissionDataManager(DataManagerUtils):
    """Data manager for the Permission model."""

    pass


class ProjectPermissionDataManager(DataManagerUtils):
    """Project Permission data manager class responsible for operations over database."""

    async def delete_project_permissions_by_user_ids(
        self, user_ids: List[UUID], project_id: UUID
    ) -> List[ProjectPermissionModel]:
        """Delete all project permissions by user ids."""

        stmt = delete(ProjectPermissionModel).where(
            ProjectPermissionModel.user_id.in_(user_ids),
            ProjectPermissionModel.project_id == project_id,
        )
        return await self.execute_commit(stmt)
