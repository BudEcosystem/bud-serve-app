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


"""Implements auth services and business logic that power the microservices, including key functionality and integrations."""

from budapp.commons import logging
from budapp.commons.constants import UserStatusEnum
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from budapp.commons.keycloak import KeycloakManager
from budapp.user_ops.crud import UserDataManager
from budapp.user_ops.models import Tenant, TenantClient, TenantUserMapping
from budapp.user_ops.models import User as UserModel
from budapp.user_ops.schemas import TenantClientSchema

from .schemas import UserLogin, UserLoginData, LogoutRequest
from .token import TokenService
from budapp.commons.config import app_settings

logger = logging.get_logger(__name__)


class AuthService(SessionMixin):
    async def login_user(self, user: UserLogin) -> UserLoginData:
        """Login a user with email and password."""
        # Get user
        db_user = await UserDataManager(self.session).retrieve_by_fields(
            UserModel, {"email": user.email}, missing_ok=True
        )

        # Check if user exists
        if not db_user:
            logger.debug(f"User not found in database: {user.email}")
            raise ClientException("This email is not registered")

        # Get tenant information
        tenant = None
        if user.tenant_id:
            tenant = await UserDataManager(self.session).retrieve_by_fields(
                Tenant, {"id": user.tenant_id}, missing_ok=True
            )
            if not tenant:
                raise ClientException("Invalid tenant ID")

            # Verify user belongs to tenant
            tenant_mapping = await UserDataManager(self.session).retrieve_by_fields(
                TenantUserMapping,
                {"tenant_id": user.tenant_id, "user_id": db_user.id},
                missing_ok=True
            )
            if not tenant_mapping:
                raise ClientException("User does not belong to this tenant")
        else:
            # If no tenant specified, get the first tenant the user belongs to
            tenant_mapping = await UserDataManager(self.session).retrieve_by_fields(
                TenantUserMapping,
                {"user_id": db_user.id},
                missing_ok=True
            )
            if tenant_mapping:
                tenant = await UserDataManager(self.session).retrieve_by_fields(
                    Tenant, {"id": tenant_mapping.tenant_id}, missing_ok=True
                )

        if not tenant:
            raise ClientException("User does not belong to any tenant")

        # Get tenant client credentials
        tenant_client = await UserDataManager(self.session).retrieve_by_fields(
            TenantClient,
            {"tenant_id": tenant.id},
            missing_ok=True
        )
        if not tenant_client:
            raise ClientException("Tenant client configuration not found")

        # Authenticate with Keycloak
        keycloak_manager = KeycloakManager()
        credentials = TenantClientSchema(
            id=tenant_client.id,
            client_id=tenant_client.client_id,
            client_secret=tenant_client.client_secret
        )

        token_data = await keycloak_manager.authenticate_user(
            username=user.email,
            password=user.password,
            realm_name=tenant.realm_name, # default realm name
            credentials=credentials
        )

        logger.debug(f"Token data: {token_data}")

        if not token_data:
            logger.debug(f"Invalid credentials for user: {user.email}")
            raise ClientException("Incorrect email or password")

        if db_user.status == UserStatusEnum.DELETED:
            logger.debug(f"User account is not active: {user.email}")
            raise ClientException("User account is not active")

        logger.debug(f"User Retrieved: {user.email}")

        # Create auth token
        # token = await TokenService(self.session).create_auth_token(str(db_user.auth_id))

        return UserLoginData(
            token=token_data,
            first_login=db_user.first_login,
            is_reset_password=db_user.is_reset_password,
        )

    async def logout_user(self, logout_data: LogoutRequest) -> None:
        """Logout a user by invalidating their refresh token."""
        # Get tenant information
        tenant = None
        if logout_data.tenant_id:
            tenant = await UserDataManager(self.session).retrieve_by_fields(
                Tenant, {"id": logout_data.tenant_id}, missing_ok=True
            )
            if not tenant:
                raise ClientException("Invalid tenant ID")
        else:
            # fetch default tenant
            tenant = await UserDataManager(self.session).retrieve_by_fields(
                Tenant, {"realm_name": app_settings.default_realm_name}, missing_ok=True
            )
            if not tenant:
                raise ClientException("Default tenant not found")

        # Get tenant client credentials
        tenant_client = await UserDataManager(self.session).retrieve_by_fields(
            TenantClient,
            {"tenant_id": tenant.id},
            missing_ok=True
        )
        if not tenant_client:
            raise ClientException("Tenant client configuration not found")

        # Logout from Keycloak
        keycloak_manager = KeycloakManager()
        credentials = TenantClientSchema(
            id=tenant_client.id,
            client_id=tenant_client.client_id,
            client_secret=tenant_client.client_secret
        )

        success = await keycloak_manager.logout_user(
            refresh_token=logout_data.refresh_token,
            realm_name=tenant.realm_name,
            credentials=credentials
        )

        if not success:
            raise ClientException("Failed to logout user")
