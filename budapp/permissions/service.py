import json
from typing import List

from fastapi import status

from budapp.auth.schemas import DeletePermissionRequest, ResourceCreate
from budapp.commons import logging
from budapp.commons.config import app_settings
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
            # Get the user
            db_user = await UserDataManager(self.session).retrieve_by_fields(UserModel, {"id": user_id})

            if not db_user:
                raise ClientException("User not found", status_code=status.HTTP_404_NOT_FOUND)

            # Keycloak Manager
            kc_manager = KeycloakManager()
            realm_admin = kc_manager.get_realm_admin(app_settings.default_realm_name)

            # Get or create user policy
            policy_name = f"urn:bud:policy:{db_user.auth_id}"
            existing_policies = realm_admin.get_client_authz_policies(str(tenant_client.client_id))
            user_policy = next((p for p in existing_policies if p["name"] == policy_name), None)

            if not user_policy:
                # Create user policy
                user_policy_data = {
                    "name": policy_name,
                    "description": f"User policy for {db_user.auth_id}",
                    "logic": "POSITIVE",
                    "users": [str(db_user.auth_id)],
                }

                policy_url = f"{app_settings.keycloak_server_url}/admin/realms/{app_settings.default_realm_name}/clients/{tenant_client.client_id}/authz/resource-server/policy/user"

                data_raw = realm_admin.connection.raw_post(
                    policy_url,
                    data=json.dumps(user_policy_data),
                    max=-1,
                    permission=False,
                )

                user_policy_id = data_raw.json()["id"]
            else:
                user_policy_id = user_policy["id"]

            # Process each permission update
            for permission in permissions:
                # Parse permission name (e.g., "cluster:view" -> module="cluster", scope="view")
                parts = permission.name.split(":")
                if len(parts) != 2:
                    logger.warning(f"Invalid permission format: {permission.name}")
                    continue

                module_name, scope_name = parts

                # Skip if not a valid module
                if module_name not in ["cluster", "model", "project", "user"]:
                    logger.warning(f"Invalid module name: {module_name}")
                    continue

                # Get the permission URL
                permission_search_url = f"{app_settings.keycloak_server_url}/admin/realms/{app_settings.default_realm_name}/clients/{tenant_client.client_id}/authz/resource-server/permission?name=urn%3Abud%3Apermission%3A{module_name}%3Amodule%3A{scope_name}&scope={scope_name}&type=scope"

                try:
                    data_raw = realm_admin.connection.raw_get(
                        permission_search_url,
                        max=-1,
                        permission=False,
                    )

                    if not data_raw.json():
                        logger.warning(f"Permission not found: {permission.name}")
                        continue

                    permission_id = data_raw.json()[0]["id"]

                    # Get current permission details
                    permission_url = f"{app_settings.keycloak_server_url}/admin/realms/{app_settings.default_realm_name}/clients/{tenant_client.client_id}/authz/resource-server/permission/scope/{permission_id}"
                    permission_data_raw = realm_admin.connection.raw_get(
                        permission_url,
                        max=-1,
                        permission=False,
                    )

                    # Get associated resources
                    permission_resources_url = f"{app_settings.keycloak_server_url}/admin/realms/{app_settings.default_realm_name}/clients/{tenant_client.client_id}/authz/resource-server/policy/{permission_id}/resources"
                    permission_resources_data_raw = realm_admin.connection.raw_get(
                        permission_resources_url,
                        max=-1,
                        permission=False,
                    )

                    # Get scopes
                    permission_scopes_url = f"{app_settings.keycloak_server_url}/admin/realms/{app_settings.default_realm_name}/clients/{tenant_client.client_id}/authz/resource-server/policy/{permission_id}/scopes"
                    permission_scopes_data_raw = realm_admin.connection.raw_get(
                        permission_scopes_url,
                        max=-1,
                        permission=False,
                    )

                    # Get associated policies
                    permission_policies_url = f"{app_settings.keycloak_server_url}/admin/realms/{app_settings.default_realm_name}/clients/{tenant_client.client_id}/authz/resource-server/policy/{permission_id}/associatedPolicies"
                    permission_policies_data_raw = realm_admin.connection.raw_get(
                        permission_policies_url,
                        max=-1,
                        permission=False,
                    )

                    # Update policy list based on has_permission
                    update_policies = []
                    for policy in permission_policies_data_raw.json():
                        if policy["name"] != policy_name:
                            update_policies.append(policy["id"])

                    if permission.has_permission:
                        # Add user policy if not already present
                        if user_policy_id not in update_policies:
                            update_policies.append(user_policy_id)
                    else:
                        # Remove user policy if present
                        if user_policy_id in update_policies:
                            update_policies.remove(user_policy_id)

                    # Update the permission
                    permission_update_payload = {
                        "id": permission_data_raw.json()["id"],
                        "name": permission_data_raw.json()["name"],
                        "description": permission_data_raw.json()["description"],
                        "type": permission_data_raw.json()["type"],
                        "logic": permission_data_raw.json()["logic"],
                        "decisionStrategy": permission_data_raw.json()["decisionStrategy"],
                        "resources": [r["_id"] for r in permission_resources_data_raw.json()],
                        "policies": update_policies,
                        "scopes": [s["id"] for s in permission_scopes_data_raw.json()],
                    }

                    # Update the permission
                    permission_update_url = f"{app_settings.keycloak_server_url}/admin/realms/{app_settings.default_realm_name}/clients/{tenant_client.client_id}/authz/resource-server/permission/scope/{permission_id}"
                    realm_admin.connection.raw_put(
                        permission_update_url,
                        data=json.dumps(permission_update_payload),
                        max=-1,
                        permission=False,
                    )

                    logger.info(f"Updated permission {permission.name} for user {user_id}")

                except Exception as e:
                    logger.error(f"Error updating permission {permission.name}: {str(e)}")
                    continue

            # Return the updated permissions
            return permissions

        except Exception as e:
            logger.error(f"Error updating global permissions: {str(e)}")
            raise
