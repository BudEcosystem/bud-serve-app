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

"""The model ops package, containing essential business logic, services, and routing configurations for the user ops."""

from typing import Union

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.commons import logging
from budapp.commons.dependencies import get_current_active_invite_user, get_session
from budapp.commons.schemas import ErrorResponse
from budapp.user_ops.schemas import User

from .schemas import UserResponse


logger = logging.get_logger(__name__)

user_router = APIRouter(prefix="/users", tags=["user"])


@user_router.get(
    "/me",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
        status.HTTP_200_OK: {
            "model": UserResponse,
            "description": "Successfully get current user",
        },
    },
    description="Get current user",
)
async def get_current_user(
    current_user: Annotated[User, Depends(get_current_active_invite_user)],
    session: Annotated[Session, Depends(get_session)],
) -> Union[UserResponse, ErrorResponse]:
    """Get current user."""
    try:
        logger.debug(f"Active user retrieved: {current_user.email}")
        return UserResponse(
            object="user.me", code=status.HTTP_200_OK, message="Successfully get current user", user=current_user
        ).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to get current user: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get current user"
        ).to_http_response()
