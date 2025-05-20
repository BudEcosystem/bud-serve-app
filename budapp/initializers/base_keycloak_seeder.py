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

        # Check if realm exists in Keycloak
        keycloak_realm_exists = keycloak_manager.realm_exists(default_realm_name)

        # Check if user exists in DB
        db_user = await UserDataManager(session).retrieve_by_fields(
            UserModel,
            {"email": app_settings.superuser_email, "status": UserStatusEnum.ACTIVE, "is_superuser": True},
            missing_ok=True,
        )

        # If both exist, we can skip the entire seeding process
        if keycloak_realm_exists and db_user:
            logger.info(f"::KEYCLOAK::Realm {default_realm_name} and user {app_settings.superuser_email} both exist. Skipping seeding.")
            return

        # Create realm in Keycloak if it doesn't exist
        if not keycloak_realm_exists:
            logger.debug(f"::KEYCLOAK::Realm {default_realm_name} does not exist. Creating...")
            await keycloak_manager.create_realm(default_realm_name)

        # Check if tenant exists in database
        tenant = await UserDataManager(session).retrieve_by_fields(
            Tenant,
            {"realm_name": default_realm_name},
            missing_ok=True,
        )

        if not tenant:
            # Save The Tenant in DB if it doesn't exist
            tenant = Tenant(
                name="Default Tenant",
                realm_name=default_realm_name,
                tenant_identifier=default_realm_name,
                description="Default tenant for superuser",
                is_active=True,
            )
            tenant = await UserDataManager(session).insert_one(tenant)
            logger.info(f"::KEYCLOAK::Tenant created in DB with ID {tenant.id}")
        else:
            logger.info(f"::KEYCLOAK::Tenant already exists in DB with ID {tenant.id}")

        # Check if the client exists for the tenant
        tenant_client = await UserDataManager(session).retrieve_by_fields(
            TenantClient,
            {"tenant_id": tenant.id, "client_named_id": default_client_id},
            missing_ok=True,
        )

        # Create client in Keycloak if realm was just created
        if not keycloak_realm_exists:
            new_client_id, client_secret = await keycloak_manager.create_client(default_client_id, default_realm_name)

            if not tenant_client:
                # Create new client record in DB
                tenant_client = TenantClient(
                    tenant_id=tenant.id,
                    client_named_id=default_client_id,
                    client_id=new_client_id,
                    client_secret=client_secret,
                )
                await UserDataManager(session).insert_one(tenant_client)
                logger.info(f"::KEYCLOAK::Client created in DB with ID {tenant_client.id}")
            else:
                # Update existing client record with new Keycloak credentials
                tenant_client.client_id = new_client_id
                tenant_client.client_secret = client_secret
                UserDataManager(session).update_one(tenant_client)
                logger.info(f"::KEYCLOAK::Client updated in DB with ID {tenant_client.id}")
        else:
            # If we get here, realm exists but user doesn't, we need to fetch client info
            if not tenant_client:
                logger.error("::KEYCLOAK::Inconsistent state: Realm exists but no client record in DB")
                return

        # If realm was just created or user doesn't exist, create user in Keycloak
        if not keycloak_realm_exists or not db_user:
            if keycloak_realm_exists and not db_user:
                logger.info(f"::KEYCLOAK::Realm exists but user {app_settings.superuser_email} doesn't exist. Creating user...")

            # Create user in Keycloak
            keycloak_user_id = await keycloak_manager.create_realm_admin(
                username=app_settings.superuser_email,
                email=app_settings.superuser_email,
                password=app_settings.superuser_password,
                realm_name=default_realm_name,
                client_id=tenant_client.client_id,
                client_secret=tenant_client.client_secret,
            )

            if not db_user:
                # Create new user record in DB
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
                db_user = await UserDataManager(session).insert_one(db_user)
                logger.info(f"::KEYCLOAK::User created in DB with ID {db_user.id}")

                # Add user to tenant mapping
                tenant_user_mapping = await UserDataManager(session).retrieve_by_fields(
                    TenantUserMapping,
                    {"tenant_id": tenant.id, "user_id": db_user.id},
                    missing_ok=True,
                )

                if not tenant_user_mapping:
                    tenant_user_mapping = TenantUserMapping(
                        tenant_id=tenant.id,
                        user_id=db_user.id,
                    )
                    await UserDataManager(session).insert_one(tenant_user_mapping)
                    logger.info("::KEYCLOAK::User-Tenant mapping created")
            else:
                # Update existing user record with new Keycloak ID
                db_user.auth_id = keycloak_user_id
                UserDataManager(session).update_one(db_user)
                logger.info("::KEYCLOAK::User updated in DB with new auth_id")

        logger.info("::KEYCLOAK::Seeding completed successfully")
