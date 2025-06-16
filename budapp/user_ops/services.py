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

from typing import Dict, List, Optional, Tuple
from uuid import UUID

from fastapi import HTTPException, status

from budapp.commons import logging
from budapp.commons.config import app_settings
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import BudNotifyException, ClientException
from budapp.commons.keycloak import KeycloakManager
from budapp.core.schemas import SubscriberCreate, SubscriberUpdate
from budapp.shared.notification_service import BudNotifyHandler
from budapp.user_ops.crud import UserDataManager
from budapp.user_ops.models import Tenant, TenantClient, TenantUserMapping
from budapp.user_ops.models import User as UserModel
from budapp.user_ops.schemas import ResetPasswordRequest, TenantClientSchema

from ..commons.constants import BUD_RESET_PASSWORD_WORKFLOW, EndpointStatusEnum, NotificationCategory, UserStatusEnum
from ..commons.helpers import generate_valid_password, validate_password_string
from ..credential_ops.crud import CredentialDataManager
from ..credential_ops.models import Credential as CredentialModel
from ..endpoint_ops.crud import EndpointDataManager
from ..endpoint_ops.models import Endpoint as EndpointModel
from ..shared.notification_service import BudNotifyService, NotificationBuilder


logger = logging.get_logger(__name__)
settings = app_settings

class UserService(SessionMixin):

    async def get_permissions_for_users(
        self,
        user_ids: List[UUID],
    ) -> Dict[str, Dict[str, List[str]]]:
        """Get permissions for users."""
        # Default Client Details
        tenant = await UserDataManager(self.session).retrieve_by_fields(
            Tenant, {"realm_name": app_settings.default_realm_name}, missing_ok=True
        )
        tenant_client = await UserDataManager(self.session).retrieve_by_fields(
            TenantClient, {"tenant_id": tenant.id}, missing_ok=True
        )

        return await KeycloakManager().get_multiple_users_permissions_via_admin(user_ids, app_settings.default_realm_name, tenant_client.client_id)

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
        db_user = await UserDataManager(self.session).retrieve_by_fields(
            UserModel, {"id": user_id}
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
                fields["first_login"] = False

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
                await BudNotifyHandler().update_subscriber(
                    str(user_id), subscriber_data
                )
                logger.info("Updated Budserve user in BudNotify subscriber")
                fields["is_subscriber"] = True
            except BudNotifyException as e:
                fields["is_subscriber"] = False
                logger.error(f"Failed to update user in budnotify subscriber: {e}")

        return await UserDataManager(self.session).update_by_fields(
            db_user, fields
        )


    async def get_user_roles_and_permissions(
        self,
        user: UserModel,
    ) -> UserModel:
        """Get user roles and permissions."""
        try:
            auth_id = user.auth_id
            if not auth_id:
                logger.error("User auth_id is missing")
                raise ClientException("User authentication ID not found", status_code=401)

            # Realm Name
            realm_name = app_settings.default_realm_name

            # Default Client Details
            tenant = await UserDataManager(self.session).retrieve_by_fields(
                Tenant, {"realm_name": realm_name}, missing_ok=True
            )
            if not tenant:
                logger.error(f"Tenant not found for realm: {realm_name}")
                raise ClientException("Tenant configuration not found", status_code=401)

            tenant_client = await UserDataManager(self.session).retrieve_by_fields(
                TenantClient, {"tenant_id": tenant.id}, missing_ok=True
            )
            if not tenant_client:
                logger.error(f"Tenant client not found for tenant: {tenant.id}")
                raise ClientException("Tenant client configuration not found", status_code=401)

            # Validate token exists
            if not user.raw_token:
                logger.error("User token is missing")
                raise ClientException("User authentication token not found", status_code=401)

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
        except ClientException:
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting user roles and permissions: {str(e)}", exc_info=True)
            raise ClientException(f"Failed to retrieve user permissions: {str(e)}", status_code=401)

    async def get_user_permissions_by_id(
        self,
        user_id: UUID,
    ) -> UserModel:
        """Get user roles and permissions."""
        # Relan Name
        realm_name = app_settings.default_realm_name

        # Get the user from db
        db_user = await UserDataManager(self.session).retrieve_by_fields(
            UserModel, {"id": user_id}, missing_ok=True
        )

        if not db_user:
            raise ClientException(message="User not found", status_code=status.HTTP_404_NOT_FOUND)

        # Default Client Details
        tenant = await UserDataManager(self.session).retrieve_by_fields(
            Tenant, {"realm_name": realm_name}, missing_ok=True
        )
        tenant_client = await UserDataManager(self.session).retrieve_by_fields(
            TenantClient, {"tenant_id": tenant.id}, missing_ok=True
        )


        # Keycloak Manager
        keycloak_manager = KeycloakManager()
        result = await keycloak_manager.get_user_permissions_via_admin(
            str(db_user.auth_id),
            realm_name,
            tenant_client.client_id,
        )

        logger.debug(f"::KEYCLOAK::User {db_user.auth_id} roles and permissions: {result}")

        return result

    async def get_all_users(
        self,
        offset: int = 0,
        limit: int = 10,
        filters: Dict = {},
        order_by: List = [],
        search: bool = False,
    ) -> Tuple[List[UserModel], int]:
        """Get all users from the database."""
        return await UserDataManager(self.session).get_all_users(offset, limit, filters, order_by, search)

    async def complete_user_onboarding(self, db_user: UserModel) -> UserModel:
        """Complete user onboarding."""
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

    async def retrieve_active_user(self, user_id: UUID) -> Optional[UserModel]:
        """Retrieve active user by id."""
        return await UserDataManager(self.session).retrieve_active_or_invited_user(user_id)

    async def delete_active_user(self, user_id: UUID, remove_credential: bool) -> UserModel:
        db_user = await UserDataManager(self.session).retrieve_by_fields(
            UserModel, {"id": user_id, "status": UserStatusEnum.ACTIVE}
        )

        if db_user.is_superuser:
            raise ClientException("Cannot delete superuser")

        # Delete active endpoints created by user
        db_endpoints = await EndpointDataManager(self.session).get_all_by_fields(
            EndpointModel, fields={"created_by": user_id}, exclude_fields={"status": EndpointStatusEnum.DELETED}
        )

        if db_endpoints:
            raise ClientException("User has active endpoints")

        # Delete user credentials
        db_credentials = await CredentialDataManager(self.session).get_all_by_fields(
            CredentialModel, fields={"user_id": db_user.id}
        )

        if db_credentials and not remove_credential:
            logger.info("Found user created credentials related to user")
            raise ClientException("Credentials need to be removed")
        else:
            # Delete all credentials related to the user
            await CredentialDataManager(self.session).delete_credential_by_fields({"user_id": db_user.id})
            logger.info("Deleted project credentials related to user")

        # Update user fields
        data = {"status": UserStatusEnum.DELETED}

        # Add user to budnotify subscribers
        try:
            await BudNotifyHandler().delete_subscriber(str(db_user.id))
            logger.info("Deleted Budserve user from BudNotify subscriber")

            # In order to prevent this user sync in periodic task, set is_subscriber to True
            data["is_subscriber"] = True
        except BudNotifyException as e:
            logger.error(f"Failed to delete user from budnotify subscribers: {e}")

        return await UserDataManager(self.session).update_by_fields(db_user, data)

    async def reactivate_user(self, user_id: UUID) -> UserModel:
        """Reactivate a user."""
        db_user = await UserDataManager(self.session).retrieve_by_fields(
            UserModel, {"id": user_id, "status": UserStatusEnum.DELETED}, missing_ok=True
        )

        if not db_user:
            raise ClientException(message="Inactive user not found", status_code=status.HTTP_404_NOT_FOUND)

        # Update user fields
        data = {"status": UserStatusEnum.ACTIVE}

        # Add user to budnotify subscribers
        try:
            subscriber_data = SubscriberCreate(
                subscriber_id=str(db_user.id),
                email=db_user.email,
                first_name=db_user.name,
            )
            await BudNotifyHandler().create_subscriber(subscriber_data)
            logger.info("Reactivated Budserve user in BudNotify subscriber")

            # In order to prevent this user sync in periodic task, set is_subscriber to True
            data["is_subscriber"] = True
        except BudNotifyException as e:
            data["is_subscriber"] = False
            logger.error(f"Failed to reactivate user in budnotify subscribers: {e}")

        return await UserDataManager(self.session).update_by_fields(db_user, data)

    async def reset_password_email(self, request: ResetPasswordRequest):
        """Trigger a reset password email notification."""
        # Check if user exists
        db_user = await UserDataManager(self.session).retrieve_by_fields(
            UserModel, {"email": request.email}, missing_ok=True
        )

        if not db_user:
            raise ClientException(message="Email not registered", status_code=status.HTTP_404_NOT_FOUND)

        # Check if user is active
        if db_user.status == UserStatusEnum.DELETED:
            raise ClientException(message="Inactive user not allowed to reset password", status_code=status.HTTP_400_BAD_REQUEST)

        # Check user is super admin
        if db_user.is_superuser:
            raise ClientException(message="Super user not allowed to reset password", status_code=status.HTTP_400_BAD_REQUEST)

        # Check max reset password attempts exceeded
        if db_user.reset_password_attempt == 3:
            raise ClientException(message="Reset password attempt limit exceeded", status_code=status.HTTP_400_BAD_REQUEST)

        # Generate a temporary password
        temp_password = ""
        while True:
            logger.debug("Generating temporary password")
            temp_password = generate_valid_password()
            is_valid, message = validate_password_string(temp_password)
            if is_valid:
                break
            logger.debug(f"Temp password invalid: {message}")

        # Get tenant information
        tenant = None
        if request.tenant_id:
            tenant = await UserDataManager(self.session).retrieve_by_fields(
                Tenant, {"id": request.tenant_id}, missing_ok=True
            )
            if not tenant:
                raise ClientException("Invalid tenant ID")

            # Verify user belongs to tenant
            tenant_mapping = await UserDataManager(self.session).retrieve_by_fields(
                TenantUserMapping, {"tenant_id": request.tenant_id, "user_id": db_user.id}, missing_ok=True
            )
            if not tenant_mapping:
                raise ClientException("User does not belong to this tenant")
        else:
            # Get the default tenant
            tenant = await UserDataManager(self.session).retrieve_by_fields(
                Tenant, {"realm_name": app_settings.default_realm_name}, missing_ok=True
            )
            if not tenant:
                raise ClientException("Default tenant not found")

        logger.debug(f"::USER:: Tenant: {tenant.realm_name}")

        if not tenant:
            raise ClientException("User does not belong to any tenant")

        # Update user password in Keycloak
        keycloak_manager = KeycloakManager()
        await keycloak_manager.update_user_password(
            db_user.auth_id,
            temp_password,
            tenant.realm_name,  # default realm name,
        )

        # Increment reset password attempt
        fields = {}
        reset_password_attempt = db_user.reset_password_attempt
        fields["reset_password_attempt"] = reset_password_attempt + 1

        # User need to change password after next login
        fields["is_reset_password"] = True

        # Update in database
        db_user = await UserDataManager(self.session).update_by_fields(db_user, fields)
        logger.debug("Temporary password updated in database")

        # Send email notification to user
        content = {"password": temp_password}
        notification_request = (
            NotificationBuilder()
            .set_content(content=content)
            .set_payload(category=NotificationCategory.INTERNAL)
            .set_notification_request(subscriber_ids=[str(db_user.id)], name=BUD_RESET_PASSWORD_WORKFLOW)
            .build()
        )
        notification_request.payload.content = content

        try:
            response = await BudNotifyService().send_notification(notification_request)
            if "object" in response and response["object"] == "error":
                logger.error(f"Failed to send notification {response}")
                raise ClientException(
                    message="Failed to send email", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                ) from None

            logger.debug(f"Sent email notification to {db_user.id}")
            return response
        except BudNotifyException as err:
            logger.error(f"Failed to send notification {err.message}")
            raise ClientException(
                message="Failed to send email", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from None
