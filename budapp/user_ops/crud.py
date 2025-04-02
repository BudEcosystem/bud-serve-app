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

"""The crud package, containing essential business logic, services, and routing configurations for the user ops."""

from typing import List
from uuid import UUID

from sqlalchemy import select, or_

from budapp.commons import logging
from budapp.commons.constants import UserStatusEnum
from budapp.commons.db_utils import DataManagerUtils

from .models import User


logger = logging.get_logger(__name__)


class UserDataManager(DataManagerUtils):
    """Data manager for the User model."""

    async def get_active_invited_users_by_ids(self, user_ids: List[UUID]) -> List[User]:
        """Get users by ids from database."""

        stmt = select(User).filter(
            User.id.in_(user_ids),
            or_(
                User.status == UserStatusEnum.ACTIVE,
                User.status == UserStatusEnum.INVITED,
            ),
        )
        return self.scalars_all(stmt)

    async def get_users_by_emails(self, emails: List[str]) -> List[User]:
        """Get users by emails from database."""

        stmt = select(User).filter(User.email.in_(emails))
        return self.scalars_all(stmt)
