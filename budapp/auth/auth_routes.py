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

"""Defines authentication routes for the microservices, providing endpoints for user authentication."""

from typing import Union

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.commons import logging
from budapp.commons.dependencies import get_session
from budapp.commons.exceptions import ClientException
from budapp.commons.schemas import ErrorResponse

from .schemas import LogoutResponse, UserLogin, UserLoginResponse, LogoutRequest
from .services import AuthService


logger = logging.get_logger(__name__)

auth_router = APIRouter(prefix="/auth", tags=["auth"])


@auth_router.post(
    "/login",
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
            "model": UserLoginResponse,
            "description": "Successfully logged in user",
        },
    },
    description="Login a user with email and password",
)
async def login_user(
    user: UserLogin, session: Annotated[Session, Depends(get_session)]
) -> Union[UserLoginResponse, ErrorResponse]:
    """Login a user with email and password."""
    try:
        auth_token = await AuthService(session).login_user(user)
        return UserLoginResponse(
            code=status.HTTP_200_OK,
            message="User logged in successfully",
            token=auth_token.token,
            first_login=auth_token.first_login,
            is_reset_password=auth_token.is_reset_password,
            object="auth_token",
        ).to_http_response()
    except ClientException as e:
        logger.error(f"ClientException: {e}")
        return ErrorResponse(code=status.HTTP_400_BAD_REQUEST, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Exception: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Something went wrong"
        ).to_http_response()

@auth_router.post(
    "/logout",
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
            "model": LogoutResponse,
            "description": "Successfully logged out user",
        },
    },
    description="Logout a user by invalidating their refresh token",
)
async def logout_user(
    logout_data: LogoutRequest, session: Annotated[Session, Depends(get_session)]
) -> Union[LogoutResponse, None]:
    """Logout a user by invalidating their refresh token."""
    try:
        await AuthService(session).logout_user(logout_data)
        return LogoutResponse(
            code=status.HTTP_200_OK,
            message="User logged out successfully"
        ).to_http_response()
    except ClientException as e:
        logger.error(f"ClientException: {e}")
        return ErrorResponse(code=status.HTTP_400_BAD_REQUEST, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Exception: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Something went wrong"
        ).to_http_response()
