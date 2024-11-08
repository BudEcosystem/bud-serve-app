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
from uuid import UUID

from fastapi import Depends, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.auth.schemas import AccessTokenData
from budapp.commons import logging
from budapp.commons.config import secrets_settings
from budapp.commons.constants import JWT_ALGORITHM, TokenTypeEnum, UserStatusEnum
from budapp.commons.database import SessionLocal
from budapp.user_ops.crud import UserDataManager
from budapp.user_ops.models import User as UserModel
from budapp.user_ops.schemas import User


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
    # Define an exception to be raised for invalid credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # Decode the token and verify its validity
        payload = jwt.decode(
            token.credentials,
            secrets_settings.jwt_secret_key,
            algorithms=[JWT_ALGORITHM],
        )

        # Raise an exception if the token is not an access token
        if payload.get("type") != TokenTypeEnum.ACCESS.value:
            raise credentials_exception from None

        # Extract the user ID from the payload. Raise an exception if it can't be found
        auth_id: str = payload.get("sub")
        if not auth_id:
            raise credentials_exception from None

        # Create AccessTokenData instance with user auth ID
        token_data = AccessTokenData(sub=auth_id)
    except JWTError:
        logger.info("Invalid access token found")
        # Raise an exception if there's an issue decoding the token
        raise credentials_exception from None

    # Retrieve the user from the database
    db_user = await UserDataManager(session).retrieve_by_fields(
        UserModel, {"auth_id": UUID(token_data.sub)}, missing_ok=True
    )

    # Raise an exception if the user is not found
    if not db_user:
        raise credentials_exception from None

    return db_user


async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    """Get the current active user.

    Args:
        current_user (User): The current user.

    Returns:
        User: The current active user.
    """
    if not current_user.is_active or current_user.status != UserStatusEnum.ACTIVE:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user


async def get_current_active_invite_user(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    """Get the current active invite user.

    Args:
        current_user (User): The current user.

    Returns:
        User: The current active invite user.
    """
    # NOTE: for invited, active user will have is_active False and status INVITED | ACTIVE
    if current_user.status == UserStatusEnum.INACTIVE:
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
