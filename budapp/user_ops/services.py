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

from typing import Dict
from uuid import UUID
from budapp.commons import logging
from budapp.commons.db_utils import SessionMixin
from budapp.commons.keycloak import KeycloakManager
from budapp.user_ops.crud import UserDataManager
from budapp.user_ops.models import Tenant, TenantClient, User as UserModel
from budapp.commons.config import app_settings
from budapp.user_ops.schemas import TenantClientSchema

logger = logging.get_logger(__name__)

class UserService(SessionMixin):
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