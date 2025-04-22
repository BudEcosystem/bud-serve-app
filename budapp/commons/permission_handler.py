from functools import wraps
from typing import Callable, List, Union

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from budapp.commons import logging
from budapp.commons.constants import PermissionEnum, UserRoleEnum
from budapp.commons.dependencies import get_current_active_user, get_session
from budapp.commons.keycloak import KeycloakManager
from budapp.user_ops.crud import UserDataManager
from budapp.user_ops.models import Tenant, TenantClient
from budapp.user_ops.schemas import TenantClientSchema, User

from .config import app_settings


logger = logging.get_logger(__name__)

def require_permissions(
    permissions: Union[PermissionEnum, List[PermissionEnum]] | None = None,
    roles: Union[UserRoleEnum, List[UserRoleEnum]] | None = None
):
    """Decorator to check if user has required Keycloak roles or authorization scopes.

    Args:
        permissions: List of PermissionEnum (mapped to Keycloak resources/scopes)
        roles: List of UserRoleEnum (mapped to realm roles)

    Example:
        @require_permissions(roles=UserRoleEnum.ADMIN)
        @require_permissions(permissions=[PermissionEnum.CLUSTER_VIEW]).
    """  # noqa: D401
    if isinstance(permissions, PermissionEnum):
        permissions = [permissions]
    if isinstance(roles, UserRoleEnum):
        roles = [roles]

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(
            current_user: User = Depends(get_current_active_user),
            session: Session = Depends(get_session),
            *args,
            **kwargs,
        ):

            logger.debug(f"::PERMISSION::Checking permissions for user: {current_user.id}")
            # Superuser shortcut
            if current_user.is_superuser:
                return await func(current_user=current_user, session=session, *args, **kwargs)

            # Keycloak Manager
            keycloak_manager = KeycloakManager()

            # TenantClientSchema
            realm_name = app_settings.default_realm_name
            tenant = await UserDataManager(session).retrieve_by_fields(
                Tenant, {"realm_name": realm_name}, missing_ok=True
            )
            tenant_client = await UserDataManager(session).retrieve_by_fields(
                TenantClient, {"tenant_id": tenant.id}, missing_ok=True
            )

            credentials = TenantClientSchema(
                id=tenant_client.id,
                client_id=tenant_client.client_id,
                client_named_id=tenant_client.client_named_id,
                client_secret=tenant_client.client_secret
            )
            openid = keycloak_manager.get_keycloak_openid_client(realm_name, credentials)

            permission_strings = []
            if permissions:
                for perm in permissions:
                    module, scope = perm.value.split(":")
                    resource = f"module_{module}"
                    permission_strings.append(f"{resource}#{scope}")
            permissions_str = ",".join(permission_strings)
            logger.debug(f"::PERMISSION:: Permission Params: {permissions_str}")
            if permissions_str:
                try:
                    # Check if user has all required permissions
                    uma_permissions = openid.uma_permissions(
                        token=current_user.raw_token,
                        permissions=permissions_str,
                        resource_server_id=credentials.client_id,
                        submit_request=False
                    )

                    logger.debug(f"::PERMISSION:: UMA Permissions: {uma_permissions}")

                    # If we get here without an exception, the user has the required permissions
                    logger.debug(f"::PERMISSION:: User {current_user.id} has the required permissions")

                except Exception as e:
                    logger.warning(f"::PERMISSION:: User {current_user.id} lacks required permissions: {str(e)}")
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Insufficient permissions for this operation"
                    )

            return await func(current_user=current_user, session=session, *args, **kwargs)

        return wrapper

    return decorator
