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

"""Contains dependency injection functions and utilities for the microservices, enabling modular and reusable components across the application."""

from collections.abc import AsyncGenerator
from typing import List

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import ExpiredSignatureError, JWTError
from keycloak import KeycloakAuthenticationError, KeycloakGetError, KeycloakInvalidTokenError
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.commons import logging
from budapp.commons.config import app_settings
from budapp.commons.constants import UserStatusEnum
from budapp.commons.database import SessionLocal
from budapp.commons.keycloak import KeycloakManager
from budapp.user_ops.crud import UserDataManager
from budapp.user_ops.models import Tenant, TenantClient
from budapp.user_ops.models import User as UserModel
from budapp.user_ops.schemas import TenantClientSchema, User


logger = logging.get_logger(__name__)

security = HTTPBearer()


async def get_session() -> AsyncGenerator[Session, None]:
    """Create and yield an Session for database operations.

    This function is a dependency that provides an Session for use in FastAPI
    route handlers. It ensures that the session is properly closed after use.

    Yields:
        Session: An asynchronous SQLAlchemy session.

    Raises:
        SQLAlchemyError: If there's an error creating or using the session.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


async def get_current_user(
    token: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    session: Session = Depends(get_session),
) -> User:
    """Get the current user.

    Args:
        token (HTTPAuthorizationCredentials): The token.
        session (Session): The database session.

    Returns:
        User: The current user.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        realm_name = app_settings.default_realm_name

        logger.debug(f"::USER:: Validating token for realm: {realm_name}")

        tenant = await UserDataManager(session).retrieve_by_fields(
            Tenant, {"realm_name": realm_name, "is_active": True}, missing_ok=True
        )

        if not tenant:
            raise credentials_exception

        logger.debug(f"::USER:: Tenant found: {tenant.id}")

        tenant_client = await UserDataManager(session).retrieve_by_fields(
            TenantClient, {"tenant_id": tenant.id}, missing_ok=True
        )

        if not tenant_client:
            raise credentials_exception

        credentials = TenantClientSchema(
            id=tenant_client.id,
            client_named_id=tenant_client.client_named_id,
            client_id=tenant_client.client_id,
            client_secret=tenant_client.client_secret,
        )

        manager = KeycloakManager()

        logger.debug(f"::USER:: Token: {token.credentials}")
        payload = await manager.validate_token(token.credentials, realm_name, credentials)
        logger.debug(f"::USER:: Token validated: {payload}")

        auth_id: str = payload.get("sub")
        if not auth_id:
            raise credentials_exception

        db_user = await UserDataManager(session).retrieve_by_fields(UserModel, {"auth_id": auth_id}, missing_ok=True)

        if not db_user:
            raise credentials_exception

        db_user.raw_token = token.credentials

        logger.debug(f"::USER:: User: {db_user.raw_token}")

        return db_user

    except ExpiredSignatureError:
        logger.warning("::USER:: Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    except (JWTError, KeycloakInvalidTokenError):
        logger.warning("::USER:: Invalid JWT or token rejected by Keycloak")
        raise credentials_exception

    except (KeycloakAuthenticationError, KeycloakGetError) as e:
        logger.error(f"::USER:: Keycloak error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service unavailable",
        )

    except Exception as e:
        logger.error(f"::USER:: Unexpected error while getting current user: {str(e)}")
        raise credentials_exception


async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    """Get the current active user.

    Args:
        current_user (User): The current user.

    Returns:
        User: The current active user.
    """
    if current_user.status != UserStatusEnum.ACTIVE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user


async def get_user_realm(current_user: Annotated[User, Depends(get_current_user)]) -> str:
    """Get the user realm.

    Args:
        current_user (User): The current user.

    Returns:
        str: The user realm.
    """
    # Note : should be updated to get the realm from the user, when start to support multi-realm

    return app_settings.default_realm_name


async def get_current_active_invite_user(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    """Get the current active invite user.

    Args:
        current_user (User): The current user.

    Returns:
        User: The current active invite user.
    """
    if current_user.status == UserStatusEnum.DELETED:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user


async def parse_ordering_fields(
    order_by: Annotated[
        str | None,
        Query(
            alias="order_by",
            description="Comma-separated list of fields. Example: field1,-field2,field3:asc,field4:desc",
        ),
    ] = None,
) -> List:
    """Parse a comma-separated list of fields with optional sorting directions and returns a list of tuples containing the field name and sorting direction.

    Args:
      order_by (Annotated[
            str | None,
            Query(
                alias="order_by",
                description="Comma-separated list of fields. Example: field1,-field2,field3:asc,field4:desc",
            ),
        ]): The `parse_ordering_fields` function takes a parameter `order_by`,
    which is a comma-separated list of fields used for ordering. Each field can
    optionally include a sorting direction (asc for ascending, desc for
    descending).

    Returns:
      A list of tuples where each tuple contains a field name and its sorting
    direction (ascending or descending) based on the input order_by string provided
    in the function parameter.
    """
    order_by_list = []

    if order_by is not None and order_by != "null":
        # Split the order_by string into individual fields
        fields = order_by.split(",")

        for field in fields:
            # Skip empty fields
            if not field.strip():
                continue

            # Split field into field name and sorting direction
            parts = field.split(":")
            field_name = parts[0].strip()

            if len(parts) == 1:
                # No sorting direction specified, default to ascending
                if field_name.startswith("-"):
                    order_by_list.append((field_name[1:], "desc"))
                else:
                    order_by_list.append((field_name, "asc"))
            else:
                # Sorting direction specified
                sort_direction = parts[1].lower().strip()
                if sort_direction == "asc" or sort_direction == "desc":
                    order_by_list.append((field_name, sort_direction))

    return order_by_list
