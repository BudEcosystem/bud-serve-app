from budapp.auth.schemas import ResourceCreate
from budapp.commons import logging
from budapp.commons.config import app_settings
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from budapp.commons.keycloak import KeycloakManager
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

    async def remove_resource_permission_by_user() -> None:
        pass
