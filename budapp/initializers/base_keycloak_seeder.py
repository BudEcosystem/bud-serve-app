from sqlalchemy import UUID
from sqlalchemy.orm import Session

from budapp.commons import logging
from budapp.commons.config import app_settings
from budapp.commons.constants import UserColorEnum, UserRoleEnum, UserStatusEnum
from budapp.commons.database import engine
from budapp.commons.keycloak import KeycloakManager
from budapp.initializers.base_seeder import BaseSeeder
from budapp.user_ops.crud import UserDataManager
from budapp.user_ops.models import Tenant, TenantClient, TenantUserMapping
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
                import traceback
                logger.error(f"Error during seeding: {traceback.format_exc()}")
                logger.error(f"Failed to complete seeding. Error: {e}")

    @staticmethod
    async def _seed_keycloak(session: Session) -> None:
        """Seed the keycloak."""
        # Get the keycloak admin client
        keycloak_manager = KeycloakManager()

        # Get the default realm name
        default_realm_name = app_settings.default_realm_name
        default_client_id = "default-internal-client"

        # check if its already exisits
        if not keycloak_manager.realm_exists(default_realm_name):
            logger.debug(f"::KEYCLOAK::Realm {default_realm_name} does not exist. Creating...")

            await keycloak_manager.create_realm(default_realm_name)

            # Save The Tenant in DB
            tenant = Tenant(
                name="Default Tenant",
                realm_name=default_realm_name,
                tenant_identifier=default_realm_name,
                description="Default tenant for superuser",
                is_active=True,
            )
            tenant = await UserDataManager(session).insert_one(tenant)

        # check if relm alreasy exisits in db
        tenant = await UserDataManager(session).retrieve_by_fields(
            Tenant,
            {"realm_name": default_realm_name},
            missing_ok=True,
        )

        if not tenant:
            # Save The Tenant in DB
            tenant = Tenant(
                name="Default Tenant",
                realm_name=default_realm_name,
                tenant_identifier=default_realm_name,
                description="Default tenant for superuser",
                is_active=True,
            )
            tenant = await UserDataManager(session).insert_one(tenant)



        logger.debug(f"::KEYCLOAK::Realm {default_realm_name} found.")
        # check if the client exists for the tenant
        tenant_client = await UserDataManager(session).retrieve_by_fields(
            TenantClient,
            {"tenant_id": tenant.id, "client_id": default_client_id},
            missing_ok=True,
        )

        #logger.debug(f"::KEYCLOAK::Client {tenant_client.id} found.")

        if not tenant_client:
            # Create the default client if it doesn't exist
            new_client_id, client_secret = await keycloak_manager.create_client(default_client_id, default_realm_name)

            # Save The Tenant Client in DB
            tenant_client = TenantClient(
                tenant_id=tenant.id,
                client_id=default_client_id,
                client_secret=client_secret,  # TODO: perform encryption before saving
            )
            await UserDataManager(session).insert_one(tenant_client)

        # check if the user exists for the tenant, make sure the user auth_id is used for keycloak id
        db_user = await UserDataManager(session).retrieve_by_fields(
            UserModel,
            {"email": app_settings.superuser_email, "status": UserStatusEnum.ACTIVE, "is_superuser": True},
            missing_ok=True,
        )

        if not db_user:
            # create super user
            keycloak_user_id = await keycloak_manager.create_realm_admin(
                username=app_settings.superuser_email,
                email=app_settings.superuser_email,
                password=app_settings.superuser_password,
                realm_name=default_realm_name,
            )

            # Save The User in DB
            db_user = UserModel(
                name="admin",
                auth_id=keycloak_user_id,
                email=app_settings.superuser_email,
                is_superuser=True,
                color=UserColorEnum.get_random_color(),
                is_reset_password=False,
                first_login=True,
                status=UserStatusEnum.ACTIVE.value,
                role=UserRoleEnum.SUPER_ADMIN.value,
            )
            await UserDataManager(session).insert_one(db_user)

            # also add to the user mapping table
            tenant_user_mapping = TenantUserMapping(
                tenant_id=tenant.id,
                user_id=db_user.id,
            )
            await UserDataManager(session).insert_one(tenant_user_mapping)

            logger.info(f"Keycloak user {app_settings.superuser_email} created with id {keycloak_user_id}")
