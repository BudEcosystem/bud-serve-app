from functools import wraps
from typing import Callable, List, Union

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from budapp.commons.constants import PermissionEnum, UserRoleEnum
from budapp.commons.dependencies import get_current_active_user, get_session
from budapp.permissions.crud import PermissionDataManager
from budapp.permissions.models import Permission as PermissionModel
from budapp.user_ops.schemas import User


def require_permissions(
    permissions: Union[PermissionEnum, List[PermissionEnum]] | None = None,
    roles: Union[UserRoleEnum, List[UserRoleEnum]] | None = None
):
    """Decorator to check if user has required permissions or roles.

    Args:
        permissions: Single permission or list of permissions required.
        roles: Single role or list of roles required.

    Example:
        @require_permissions(PermissionEnum.ENDPOINT_VIEW)
        async def my_route():
            ...

        @require_permissions([PermissionEnum.ENDPOINT_VIEW, PermissionEnum.ENDPOINT_MANAGE])
        async def my_route():
            ...

        @require_permissions(roles=UserRoleEnum.ADMIN)
        async def admin_route():
            ...
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
            # Super admin bypass
            if current_user.is_superuser:
                return await func(current_user=current_user, session=session, *args, **kwargs)

            # if role is provded , check if the user belogs to the group 

            # Role check
            # if roles and current_user.role not in [r.value for r in roles]:
            #     raise HTTPException(
            #         status_code=status.HTTP_403_FORBIDDEN,
            #         detail=f"Missing required role: {current_user.role}"
            #     )

            # # Permission check
            # if permissions:
            #     user_permissions = await PermissionDataManager(session).retrieve_by_fields(
            #         PermissionModel,
            #         {"user_id": current_user.id, "auth_id": current_user.auth_id},
            #         missing_ok=True
            #     )

            #     if not user_permissions:
            #         raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User has no permissions")

            #     missing_permissions = [
            #         p.value for p in permissions if p.value not in user_permissions.scopes_list
            #     ]

            #     if missing_permissions:
            #         raise HTTPException(
            #             status_code=status.HTTP_403_FORBIDDEN,
            #             detail=f"Missing required permissions: {', '.join(missing_permissions)}",
            #         )

            return await func(current_user=current_user, session=session, *args, **kwargs)

        return wrapper

    return decorator