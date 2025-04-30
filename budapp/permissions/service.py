from budapp.auth.schemas import DeletePermissionRequest, ResourceCreate
from budapp.commons import logging
from budapp.commons.config import app_settings
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from budapp.commons.keycloak import KeycloakManager
from budapp.permissions.schemas import CheckUserResourceScope
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
