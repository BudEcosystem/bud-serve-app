import json
from typing import List

from fastapi import status

from budapp.auth.schemas import DeletePermissionRequest, ResourceCreate
from budapp.commons import logging
from budapp.commons.config import app_settings
from budapp.commons.constants import PermissionEnum, UserStatusEnum
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from budapp.commons.keycloak import KeycloakManager
from budapp.permissions.schemas import CheckUserResourceScope, PermissionList
from budapp.user_ops.crud import UserDataManager
from budapp.user_ops.models import Tenant, TenantClient
from budapp.user_ops.models import User as UserModel
from budapp.user_ops.schemas import TenantClientSchema


logger = logging.get_logger(__name__)


class PermissionService(SessionMixin):
    """Keycloak Permission Service."""

    async def create_resource_permission_by_user(self, user: UserModel, resource: ResourceCreate) -> None:
        """Create a resource permission for a user."""
        # Get the default tenant
        tenant = await UserDataManager(self.session).retrieve_by_fields(
            Tenant, {"realm_name": app_settings.default_realm_name}, missing_ok=True
        )

        # Get the default tenant client
        tenant_client = await UserDataManager(self.session).retrieve_by_fields(
            TenantClient, {"tenant_id": tenant.id}, missing_ok=True
        )

        logger.debug(f"::PERMISSION:: Tenant client: {tenant_client.client_named_id}")

        if not tenant_client:
            raise ClientException("Tenant client not found")

        try:
            # Keycloak Manager
            kc_manager = KeycloakManager()
            _ = await kc_manager.create_resource_with_permissions(
                realm_name=app_settings.default_realm_name,
                client_id=str(tenant_client.client_id),
                resource=resource,
                user_auth_id=user.auth_id,
            )
        except Exception as e:
            logger.error(f"Error creating resource permission: {e}")
            raise


    async def delete_permission_for_resource(self, resource: DeletePermissionRequest) -> None:
        """Delete a resource permission for a user."""
        # Get the default tenant
        tenant = await UserDataManager(self.session).retrieve_by_fields(
            Tenant, {"realm_name": app_settings.default_realm_name}, missing_ok=True
        )

        tenant_client = await UserDataManager(self.session).retrieve_by_fields(
            TenantClient, {"tenant_id": tenant.id}, missing_ok=True
        )

        if not tenant_client:
            raise ClientException("Tenant client not found")

        try:
            # Keycloak Manager
            kc_manager = KeycloakManager()
            _ = await kc_manager.delete_permission_for_resource(
                realm_name=app_settings.default_realm_name,
                client_id=str(tenant_client.client_id),
                resource_type=resource.resource_type,
                resource_id=resource.resource_id,
                delete_resource=resource.delete_resource,
            )
        except Exception as e:
            logger.error(f"Error deleting resource permission: {e}")
            raise

    async def check_resource_permission_by_user(self, user: UserModel, payload: CheckUserResourceScope) -> bool:
        """Check if a user has a resource permission."""
        logger.debug(f"::PERMISSION::Checking permissions for user: {user.raw_token}")

        # if user.is_superuser:
        #     return True

        # Keycloak Manager
        keycloak_manager = KeycloakManager()

        # TenantClientSchema
        realm_name = app_settings.default_realm_name
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

        try:
            result = await keycloak_manager.get_user_roles_and_permissions(
                user.auth_id,
                realm_name,
                tenant_client.client_id,
                credentials,
                user.raw_token,
            )

            logger.debug(f"::KEYCLOAK::User {user.auth_id} roles and permissions: {result}")

            if payload.entity_id is None:
                # Global Permission
                for permission in result["permissions"]:
                    if (
                        permission["rsname"] == f"module_{payload.resource_type}"
                        and payload.scope in permission["scopes"]
                    ):
                        return True
            else:
                # Individual Resource Permission
                for permission in result["permissions"]:
                    if (
                        permission["rsname"] == f"URN::{payload.resource_type}::{payload.entity_id}"
                        and payload.scope in permission["scopes"]
                    ):
                        return True

            return False
        except Exception as e:
            logger.error(f"Error checking resource permission: {e}")
            raise

    async def remove_user_from_project_permissions(self, user: UserModel, project_id: str) -> None:
        """Remove a user from a project's permissions in Keycloak.

        This method removes the user's access to a specific project by removing
        their policy from all project permissions.

        Args:
            user: The user model to remove from the project
            project_id: The project UUID
        """
        # Get the default tenant
        tenant = await UserDataManager(self.session).retrieve_by_fields(
            Tenant, {"realm_name": app_settings.default_realm_name}, missing_ok=True
        )

        if not tenant:
            raise ClientException("Default tenant not found")

        # Get the default tenant client
        tenant_client = await UserDataManager(self.session).retrieve_by_fields(
            TenantClient, {"tenant_id": tenant.id}, missing_ok=True
        )

        if not tenant_client:
            raise ClientException("Tenant client not found")

        try:
            # Use KeycloakManager to remove user policy from project permissions
            kc_manager = KeycloakManager()
            await kc_manager.remove_user_policy_from_resource_permissions(
                realm_name=app_settings.default_realm_name,
                client_id=str(tenant_client.client_id),
                resource_type="project",
                resource_id=str(project_id),
                user_auth_id=str(user.auth_id),
            )
            logger.debug(f"Removed user {user.auth_id} from project {project_id} permissions")
        except Exception as e:
            logger.error(f"Error removing user from project permissions: {e}")
            raise ClientException("Failed to remove user from project permissions")

    async def add_user_to_project_permissions(
        self, user: UserModel, project_id: str, scopes: List[str] = None
    ) -> None:
        """Add a user to an existing project's permissions in Keycloak.

        This method is used when adding users to an existing project. The project
        resource and permissions already exist in Keycloak, so we only need to
        associate the user's policy with them.

        Args:
            user: The user model to add to the project
            project_id: The project UUID
            scopes: List of scopes to grant (defaults to ["view", "manage"])
        """
        # Get the default tenant
        tenant = await UserDataManager(self.session).retrieve_by_fields(
            Tenant, {"realm_name": app_settings.default_realm_name}, missing_ok=True
        )

        if not tenant:
            raise ClientException("Default tenant not found")

        # Get the default tenant client
        tenant_client = await UserDataManager(self.session).retrieve_by_fields(
            TenantClient, {"tenant_id": tenant.id}, missing_ok=True
        )

        if not tenant_client:
            raise ClientException("Tenant client not found")

        try:
            # Extract actual scopes from format like "endpoint:view" -> "view"
            if scopes:
                extracted_scopes = []
                for scope in scopes:
                    if ":" in scope:
                        # Split by ':' and take the last part (view/manage)
                        extracted_scopes.append(scope.split(":")[-1])
                    else:
                        # If no ':', use as is
                        extracted_scopes.append(scope)

                # Remove duplicates and filter to valid scopes
                valid_scopes = {"view", "manage"}
                keycloak_scopes = list(set(extracted_scopes) & valid_scopes)

                if not keycloak_scopes:
                    logger.warning(f"No valid scopes found in {scopes}, using defaults")
                    keycloak_scopes = ["view", "manage"]

                # If user has manage permission, ensure they also have view permission
                if "manage" in keycloak_scopes and "view" not in keycloak_scopes:
                    keycloak_scopes.append("view")
            else:
                keycloak_scopes = ["view", "manage"]

            # Use KeycloakManager to add user policy to existing project permissions
            kc_manager = KeycloakManager()
            await kc_manager.add_user_policy_to_resource_permissions(
                realm_name=app_settings.default_realm_name,
                client_id=str(tenant_client.client_id),
                resource_type="project",
                resource_id=str(project_id),
                user_auth_id=str(user.auth_id),
                scopes=keycloak_scopes,
            )
            logger.debug(
                f"Added user {user.auth_id} to project {project_id} permissions with scopes {keycloak_scopes}"
            )
        except Exception as e:
            logger.error(f"Error adding user to project permissions: {e}")
            raise ClientException("Failed to add user to project permissions")

    async def update_global_permissions(self, user_id: str, permissions: List[PermissionList]) -> List[PermissionList]:
        """Update global permissions for a specific user.

        Args:
            user_id: The ID of the user to update permissions for
            permissions: List of permissions to update

        Returns:
            List[PermissionList]: Updated list of permissions
        """
        # Get the default tenant
        tenant = await UserDataManager(self.session).retrieve_by_fields(
            Tenant, {"realm_name": app_settings.default_realm_name}, missing_ok=True
        )

        if not tenant:
            raise ClientException("Default tenant not found")

        # Get the default tenant client
        tenant_client = await UserDataManager(self.session).retrieve_by_fields(
            TenantClient, {"tenant_id": tenant.id}, missing_ok=True
        )

        if not tenant_client:
            raise ClientException("Tenant client not found")

        try:
            # Get the user - only retrieve active users
            db_user = await UserDataManager(self.session).retrieve_by_fields(
                UserModel, {"id": user_id, "status": UserStatusEnum.ACTIVE}
            )

            # Restrict update permission on super user
            if db_user.is_superuser:
                raise ClientException(
                    "Cannot update permissions for super user", status_code=status.HTTP_400_BAD_REQUEST
                )

            # Validate permissions against PermissionEnum
            valid_permissions = PermissionEnum.get_global_permissions()
            for permission in permissions:
                if permission.name not in valid_permissions:
                    raise ClientException(
                        f"Invalid permission: {permission.name}.", status_code=status.HTTP_400_BAD_REQUEST
                    )

            # Process permissions to add implicit view permissions for manage permissions
            permission_dict = {p.name: p for p in permissions}
            manage_to_view_mapping = PermissionEnum.get_manage_to_view_mapping()

            # Add implicit view permissions for manage permissions
            for permission in permissions:
                if permission.has_permission and permission.name in manage_to_view_mapping:
                    view_permission_name = manage_to_view_mapping[permission.name]
                    # Explicitly upsert the view permission
                    permission_dict[view_permission_name] = PermissionList(
                        name=view_permission_name, has_permission=True
                    )
                    logger.debug("Upsert %s for %s", view_permission_name, permission.name)

            processed_permissions = list(permission_dict.values())

            # Use KeycloakManager to update permissions
            kc_manager = KeycloakManager()
            await kc_manager.update_user_global_permissions(
                user_auth_id=db_user.auth_id,
                permissions=[p.model_dump() for p in processed_permissions],
                realm_name=app_settings.default_realm_name,
                client_id=str(tenant_client.client_id),
            )

            logger.info(f"Updated global permissions for user {user_id}")

            # Return the updated permissions
            return processed_permissions

        except ClientException:
            raise
        except Exception as e:
            logger.error(f"Error updating global permissions: {str(e)}")
            raise ClientException(
                "Failed to update global permissions", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
