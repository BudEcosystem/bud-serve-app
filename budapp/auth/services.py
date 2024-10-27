from fastapi import HTTPException, status

from budapp.commons import logging
from budapp.commons.config import secrets_settings
from budapp.commons.constants import UserStatusEnum
from budapp.commons.db_utils import SessionMixin
from budapp.commons.security import HashManager
from budapp.user_ops.crud import UserDataManager
from budapp.user_ops.models import User as UserModel

from .schemas import UserLogin, UserLoginData
from .token import TokenService


logger = logging.get_logger(__name__)


class AuthService(SessionMixin):
    async def login_user(self, user: UserLogin) -> UserLoginData:
        """Login a user with email and password."""
        # Get user
        db_user = await UserDataManager(self.session).retrieve_by_fields(
            UserModel, {"email": user.email}, missing_ok=True
        )

        # Check if user exists
        if not db_user:
            logger.debug(f"User not found in database: {user.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This email is not registered",
            )

        # Check if password is correct
        salted_password = user.password + secrets_settings.password_salt
        if not await HashManager().verify_hash(salted_password, db_user.password):
            logger.debug(f"Password incorrect for {user.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect email or password",
            )

        if db_user.status == UserStatusEnum.INACTIVE:
            logger.debug(f"User account is not active: {user.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User account is not active",
            )

        logger.debug(f"User Retrieved: {user.email}")

        # Create auth token
        token = await TokenService(self.session).create_auth_token(str(db_user.auth_id))

        return UserLoginData(
            token=token,
            first_login=db_user.first_login,
            is_reset_password=db_user.is_reset_password,
        )
