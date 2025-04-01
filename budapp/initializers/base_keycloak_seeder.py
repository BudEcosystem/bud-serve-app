from sqlalchemy import UUID
from sqlalchemy.orm import Session

from budapp.commons import logging
from budapp.commons.config import app_settings
from budapp.commons.constants import UserColorEnum, UserRoleEnum, UserStatusEnum
from budapp.commons.database import engine
from budapp.commons.keycloak import KeycloakManager
from budapp.initializers.base_seeder import BaseSeeder
from budapp.user_ops.crud import UserDataManager
from budapp.user_ops.models import User as UserModel


logger = logging.get_logger(__name__)

class BaseKeycloakSeeder(BaseSeeder):
    """Base class for keycloak seeder."""

    async def seed(self):
        """Seed the keycloak."""
        with Session(engine) as session:
            try:
                await self._seed_keycloak(session)
            except Exception as e:
                logger.error(f"Failed to create super user. Error: {e}")

    @staticmethod
    async def _seed_keycloak(session: Session) -> None:
        """Seed the keycloak."""
        # Get the keycloak admin client
        keycloak_manager = KeycloakManager()

        # Get the default realm name
        default_realm_name = app_settings.default_realm_name

        # check if its already exisits
        if not keycloak_manager.realm_exists(default_realm_name):
            await keycloak_manager.create_realm(default_realm_name)

        # # Create the default client if it doesn't exist
        # await keycloak_manager.create_client(default_realm_name)
        # above is skipped as we don't have organizations implemented yet , required when in multi-tenant mode and save credentials with map

        # create super user
        keycloak_user_id = await keycloak_manager.create_realm_admin(
            username=app_settings.superuser_email,
            email=app_settings.superuser_email,
            password=app_settings.superuser_password,
            realm_name=default_realm_name
        )

        logger.info(f"Keycloak user {app_settings.superuser_email} created with id {keycloak_user_id}")

        # Create / Update the user in the database
        db_user = await UserDataManager(session).retrieve_by_fields(
            UserModel,
            {"email": app_settings.superuser_email, "status": UserStatusEnum.ACTIVE, "is_superuser": True},
            missing_ok=True,
        )

        if not db_user:
            # Create super user
            super_user = UserModel(
                name="admin",
                auth_id=UUID(keycloak_user_id),
                email=app_settings.superuser_email,
                #password=hashed_password, #TODO: remove password field as we are using keycloak
                is_superuser=True,
                color=UserColorEnum.get_random_color(),
                is_reset_password=False,
                first_login=True,
                status=UserStatusEnum.ACTIVE.value,
                role=UserRoleEnum.SUPER_ADMIN.value,
            )
            db_user = await UserDataManager(session).insert_one(super_user)
            logger.debug("Inserted super user in database")

        pass
