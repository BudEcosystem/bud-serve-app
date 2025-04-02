# budapp/commons/keycloak.py

from typing import Optional, Tuple

from keycloak import KeycloakAdmin, KeycloakAuthenticationError, KeycloakOpenID

from budapp.commons.constants import UserRoleEnum
from budapp.user_ops.schemas import  TenantClientSchema

from . import logging
from .config import app_settings


logger = logging.get_logger(__name__)


class KeycloakManager:
    """Class to manage Keycloak operations including realms, clients, and users."""

    _instance: Optional["KeycloakManager"] = None
    _admin_client: Optional[KeycloakAdmin] = None

    def __new__(cls) -> "KeycloakManager":
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the Keycloak manager with a singleton admin client."""
        if self._admin_client is None:
            self._admin_client = KeycloakAdmin(
                server_url=app_settings.keycloak_server_url,
                username=app_settings.keycloak_admin_username,
                password=app_settings.keycloak_admin_password,
                realm_name=app_settings.keycloak_realm_name,
                user_realm_name=app_settings.keycloak_realm_name,
                verify=app_settings.keycloak_verify_ssl,
            )

    @property
    def admin_client(self) -> KeycloakAdmin:
        """Get the admin client instance."""
        return self._admin_client

    def _get_base_url(self) -> str:
        """Get the base URL based on environment."""
        protocol = "https" if app_settings.env != "development" else "http"
        return f"{protocol}://{app_settings.api_root}"

    def get_realm_admin(self, realm_name: str) -> KeycloakAdmin:
        """Get the realm admin client.

        Args:
            realm_name: Name of the realm

        Returns:
            KeycloakAdmin instance for the specified realm
        """
        try:
            return KeycloakAdmin(
                server_url=app_settings.keycloak_server_url,
                username=app_settings.keycloak_admin_username,
                password=app_settings.keycloak_admin_password,
                realm_name=realm_name,
                user_realm_name=app_settings.keycloak_realm_name,
                verify=app_settings.keycloak_verify_ssl,
            )
        except Exception as e:
            logger.error(f"Failed to create realm admin for realm {realm_name}: {str(e)}")
            raise
    def get_keycloak_openid_client(self, realm_name: str, credentials: TenantClientSchema) -> KeycloakAdmin:
        """Get the keycloak openid client.

        Args:
            realm_name: Name of the realm
        """
        return KeycloakOpenID(
            server_url=app_settings.keycloak_server_url,
            client_id=credentials.client_id,
            realm_name=realm_name,
            client_secret_key=credentials.client_secret,
        )
    async def create_realm(self, realm_name: str) -> dict:
        """Create a new realm in Keycloak.

        Args:
            realm_name: Name of the realm to create

        Returns:
            dict: Created realm representation
        """
        realm_name = realm_name.lower()
        realm_representation = {
            "realm": realm_name,
            "enabled": True,
            "sslRequired": "external",
            "registrationAllowed": False,
            "loginWithEmailAllowed": True,
            "duplicateEmailsAllowed": False,
            "resetPasswordAllowed": True,
            "editUsernameAllowed": False,
            "bruteForceProtected": True,
        }

        try:
            self.admin_client.create_realm(payload=realm_representation, skip_exists=True)
            logger.info(f"Realm {realm_name} created successfully")

            # MAP Roles
            relam_admin = self.get_realm_admin(realm_name)
            roles = [
                UserRoleEnum.ADMIN.value,
                UserRoleEnum.DEVELOPER.value,
                UserRoleEnum.TESTER.value,
                UserRoleEnum.DEVOPS.value,
                UserRoleEnum.SUPER_ADMIN.value,
            ]
            for role in roles:
                relam_admin.create_realm_role({"name": role, "description": f"Organization {role}"}, skip_exists=True)

        except Exception as e:
            logger.error(f"Failed to create realm {realm_name}: {str(e)}")
            raise

    async def create_client(self, client_id: str, realm_name: str) -> Tuple[str, str]:
        """Create a new client in Keycloak.

        Args:
            client_id: ID of the client to create
            realm_name: Name of the realm to create the client in

        Returns:
            Tuple[str, str]: Tuple containing (client_id, client_secret)
        """
        
        base_url = self._get_base_url()
        client_representation = {
            "clientId": client_id,
            "enabled": True,
            "protocol": "openid-connect",
            "publicClient": False,
            "redirectUris": [f"{base_url}/*"],
            "webOrigins": [base_url],
        }

        try:
            
            realm_admin = self.get_realm_admin(realm_name)
            new_client_id = realm_admin.create_client(payload=client_representation)
            client_secret = realm_admin.get_client_secrets(new_client_id)["value"]
            return new_client_id, client_secret
        except Exception as e:
            logger.error(f"Error creating client {client_id} in realm {realm_name}: {str(e)}")
            raise

    async def create_user(
        self, username: str, email: str, password: str, realm_name: str, role: UserRoleEnum
    ) -> str:
        """Create a new user in Keycloak.

        Args:
            username: Username of the user to create
            email: Email of the user to create
            password: Password of the user to create
            realm_name: Name of the realm to create the user in
            role: Role of the user to create
        Returns:
            str: User ID of the created user
        """
        user_representation = {
            "username": username,  # email is treated as username
            "email": email,
            "enabled": True,
            "emailVerified": True,
            "credentials": [{"type": "password", "value": password, "temporary": False}],
        }

        try:
            realm_admin = self.get_realm_admin(realm_name)
            user_id = realm_admin.create_user(payload=user_representation)
            logger.info(f"User {username} created successfully")

            # Assign role to user
            realm_admin.assign_realm_roles(user_id=user_id, roles=[role.value])
            logger.info(f"Role {role.value} assigned to user {username}")

            return user_id
        except Exception as e:
            logger.error(f"Error creating user {username} in realm {realm_name}: {str(e)}")
            raise

    async def create_realm_admin(self, username: str, email: str, password: str, realm_name: str) -> str:
        """Create a new realm admin user in Keycloak.

        Args:
            username: Username of the realm admin to create
            email: Email of the realm admin to create
            password: Password of the realm admin to create
            realm_name: Name of the realm to create the realm admin in

        Returns:
            str: User ID of the created realm admin
        """
        user_representation = {
            "username": username,
            "email": email,
            "enabled": True,
            "emailVerified": True,
            "credentials": [{"type": "password", "value": password, "temporary": False}],
        }

        try:
            realm_admin = self.get_realm_admin()
            user_id = realm_admin.create_user(payload=user_representation, exist_ok=True)
            logger.info(f"Realm admin {username} created successfully")

            # Assign role to user
            realm_admin.assign_realm_roles(user_id=user_id, roles=[UserRoleEnum.ADMIN.value])
            logger.info(f"Role {UserRoleEnum.ADMIN.value} assigned to realm admin {username}")

            return user_id
        except Exception as e:
            logger.error(f"Error creating realm admin {username} in realm {realm_name}: {str(e)}")
            raise

    def realm_exists(self, realm_name: str) -> bool:
        """Check if a realm exists in Keycloak.

        Args:
            realm_name: Name of the realm to check

        Returns:
            bool: True if the realm exists, False otherwise
        """
        try:
            realms = self.admin_client.get_realms()
            return any(realm["realm"] == realm_name for realm in realms)
        except Exception as e:
            logger.error(f"Error checking if realm {realm_name} exists: {str(e)}")
            return False
    
    async def authenticate_user(self, username: str, password: str, realm_name: str, credentials: TenantClientSchema) -> str:
        """Authenticate a user in Keycloak.

        Args:
            username: Username of the user to authenticate
            password: Password of the user to authenticate
            realm_name: Name of the realm to authenticate the user in

        Returns:
            str: User ID of the authenticated user
        """
        try:
            self._openid_client.token(username, password)
            return True
        except KeycloakAuthenticationError:
            logger.warning(f"Invalid credentials for user {username}")
            return False
        except Exception as e:
            logger.error(f"Error verifying password for user {username}: {str(e)}")
            return False
