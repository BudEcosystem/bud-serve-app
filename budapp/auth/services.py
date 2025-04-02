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


"""Implements auth services and business logic that power the microservices, including key functionality and integrations."""

from budapp.commons import logging
from budapp.commons.config import secrets_settings
from budapp.commons.constants import UserStatusEnum, UserColorEnum, PermissionEnum
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException
from budapp.commons.security import HashManager
from budapp.user_ops.crud import UserDataManager
from budapp.user_ops.models import User as UserModel
from budapp.user_ops.schemas import UserCreate
from .schemas import UserLogin, UserLoginData
from .token import TokenService
from ..permissions.crud import PermissionDataManager
from ..permissions.models import Permission as PermissionModel
from ..permissions.schemas import PermissionCreate
from ..core.schemas import SubscriberCreate
from ..commons.exceptions import BudNotifyException
from ..shared.notification_service import BudNotifyHandler

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
            raise ClientException("This email is not registered")

        # Check if password is correct
        salted_password = user.password + secrets_settings.password_salt
        if not await HashManager().verify_hash(salted_password, db_user.password):
            logger.debug(f"Password incorrect for {user.email}")
            raise ClientException("Incorrect email or password")

        if db_user.status == UserStatusEnum.DELETED:
            logger.debug(f"User account is not active: {user.email}")
            raise ClientException("User account is not active")

        logger.debug(f"User Retrieved: {user.email}")

        # Create auth token
        token = await TokenService(self.session).create_auth_token(str(db_user.auth_id))

        return UserLoginData(
            token=token,
            first_login=db_user.first_login,
            is_reset_password=db_user.is_reset_password,
        )

    async def register_user(self, user: UserCreate) -> UserModel:
        # Check if email is already registered
        email_exists = await UserDataManager(self.session).retrieve_by_fields(
            UserModel, {"email": user.email}, missing_ok=True
        )

        # Raise exception if email is already registered
        if email_exists:
            logger.info(f"Email already registered: {user.email}")
            raise ClientException(detail="Email already registered")

        # Hash password
        salted_password = user.password + secrets_settings.password_salt
        user.password = await HashManager().get_hash(salted_password)
        logger.info(f"Password hashed for {user.email}")

        user_data = user.model_dump(exclude={"permissions"})
        user_data["color"] = UserColorEnum.get_random_color()

        user_data["status"] = UserStatusEnum.INVITED

        user_model = UserModel(**user_data)

        # NOTE: is_reset_password, first_login will be set to True by default
        # NOTE: status wil be invited by default
        # Create user
        db_user = await UserDataManager(self.session).insert_one(user_model)

        # Ensure that both given scopes and default scopes are uniquely added to the user without duplication.
        scopes = PermissionEnum.get_default_permissions()
        if user.permissions:
            new_scopes = [permission.name for permission in user.permissions if permission.has_permission]
            scopes.extend(new_scopes)
        scopes = list(set(scopes))
        logger.info(f"Scopes created for {user.email}: {scopes}")

        permissions = PermissionCreate(
            user_id=db_user.id,
            auth_id=db_user.auth_id,
            scopes=scopes,
        )
        permission_model = PermissionModel(**permissions.model_dump())
        _ = await PermissionDataManager(self.session).insert_one(permission_model)

        # Add user to budnotify subscribers
        try:
            subscriber_data = SubscriberCreate(
                subscriber_id=str(db_user.id),
                email=db_user.email,
                first_name=db_user.name,
            )
            response = await BudNotifyHandler().create_subscriber(subscriber_data)
            logger.info("User added to budnotify subscriber")

            _ = await UserDataManager(self.session).update_subscriber_status(user_ids=[db_user.id], is_subscriber=True)
        except BudNotifyException as e:
            logger.error(f"Failed to add user to budnotify subscribers: {e}")

        return db_user
