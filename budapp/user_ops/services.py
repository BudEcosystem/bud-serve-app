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

"""The user ops package, containing essential business logic, services, and routing configurations for the user ops."""

from typing import Dict, List, Tuple
from uuid import UUID

from fastapi import HTTPException, status

from budapp.commons import logging
from budapp.commons.exceptions import ClientException
from budapp.commons.config import app_settings
from budapp.commons.constants import UserStatusEnum
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import BudNotifyException
from budapp.commons.keycloak import KeycloakManager
from budapp.core.schemas import SubscriberUpdate
from budapp.shared.notification_service import BudNotifyHandler
from budapp.user_ops.crud import UserDataManager
from budapp.user_ops.models import Tenant, TenantClient
from budapp.user_ops.models import User as UserModel
from budapp.user_ops.schemas import TenantClientSchema


logger = logging.get_logger(__name__)
settings = app_settings

class UserService(SessionMixin):

    async def update_active_user(
        self,
        user_id: UUID,
        fields: Dict,
        current_user: UserModel,
        realm_name: str,
    ) -> UserModel:
        """Update active user."""
        if user_id == current_user.id:  # noqa: SIM102
            # Invited users can only update password
            if current_user.status == UserStatusEnum.INVITED and {"role", "name"} & set(
                fields
            ):
                logger.error("Invited user can only update password")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invited user can only update password",
                )

        # Check if user exists
        db_user = await UserDataManager(self.session).retrieve_user_by_fields(
            {"id": user_id}
        )

        if db_user.is_superuser and "role" in fields:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot update superuser role",
            )

        if db_user.status == UserStatusEnum.DELETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot update inactive user",
            )

        if "password" in fields:
            # Keycloak Manager
            keycloak_manager = KeycloakManager()
            await keycloak_manager.update_user_password(
                current_user.auth_id,
                fields["password"],
                realm_name,
            )


            if user_id == current_user.id:
                # Updating user own password doesn't require password after login
                fields["is_reset_password"] = False

                if db_user.first_login and db_user.status == UserStatusEnum.INVITED:
                    fields["status"] = UserStatusEnum.ACTIVE
                    logger.info(f"User {user_id} status set to active")
            else:
                fields["is_reset_password"] = True

            # Reset password limit set to zero
            fields["reset_password_attempt"] = 0

        if "name" in fields:
            try:
                subscriber_data = SubscriberUpdate(
                    email=db_user.email,
                    first_name=fields.get("name"),
                )
                response = await BudNotifyHandler().update_subscriber(
                    str(user_id), subscriber_data
                )
                logger.info("Updated Budserve user in BudNotify subscriber")
                fields["is_subscriber"] = True
            except BudNotifyException as e:
                fields["is_subscriber"] = False
                logger.error(f"Failed to update user in budnotify subscriber: {e}")

        return await UserDataManager(self.session).update_user_by_fields(
            db_user, fields
        )


    async def get_user_roles_and_permissions(
        self,
        user: UserModel,
    ) -> UserModel:
        """Get user roles and permissions."""
        auth_id = user.auth_id

        # Relan Name
        realm_name = app_settings.default_realm_name

        # Default Client Details
        tenant = await UserDataManager(self.session).retrieve_by_fields(
            Tenant, {"realm_name": realm_name}, missing_ok=True
        )
        tenant_client = await UserDataManager(self.session).retrieve_by_fields(
            TenantClient, {"tenant_id": tenant.id}, missing_ok=True
        )

        # Credentials
        credentials = TenantClientSchema(
            id=tenant_client.id,
            client_named_id=tenant_client.client_named_id,
            client_id=tenant_client.client_id,
            client_secret=tenant_client.client_secret,
        )

        # Keycloak Manager
        keycloak_manager = KeycloakManager()
        result = await keycloak_manager.get_user_roles_and_permissions(
            auth_id,
            realm_name,
            tenant_client.client_id,
            credentials,
            user.raw_token,
        )

        logger.debug(f"::KEYCLOAK::User {auth_id} roles and permissions: {result}")

        return result

    async def get_all_users(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[UserModel], int]:
        """Get all users from the database"""
        return await UserDataManager(self.session).get_all_users(offset, limit, filters, order_by, search)

    async def complete_user_onboarding(self, db_user: UserModel) -> UserModel:
        """Complete user onboarding"""
        if db_user.status == UserStatusEnum.DELETED or db_user.status == UserStatusEnum.INVITED:
            raise ClientException(
                "Only active users can complete onboarding",
            )

        if not db_user.first_login:
            raise ClientException(
                "User already completed onboarding",
            )

        db_user = await UserDataManager(self.session).update_by_fields(db_user, {"first_login": False})

        return db_user
