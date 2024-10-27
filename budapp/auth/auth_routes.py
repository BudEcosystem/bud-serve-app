from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.commons import logging
from budapp.commons.dependencies import get_session
from budapp.commons.schemas import ErrorResponse

from .schemas import UserLogin, UserLoginResponse
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
async def login_user(user: UserLogin, session: Annotated[Session, Depends(get_session)]) -> UserLoginResponse:
    """Login a user with email and password."""
    auth_token = await AuthService(session).login_user(user)
    logger.debug(f"Token created for {user.email}")

    return UserLoginResponse(
        code=status.HTTP_200_OK,
        message="User logged in successfully",
        token=auth_token.token,
        first_login=auth_token.first_login,
        is_reset_password=auth_token.is_reset_password,
        object="auth_token",
    )
