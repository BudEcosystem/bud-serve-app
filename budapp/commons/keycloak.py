# budapp/commons/keycloak.py

from keycloak import KeycloakAdmin

from . import logging
from .config import app_settings


logger = logging.get_logger(__name__)


class KeycloakManager:
    """Class to manage Keycloak operations including realms, clients, and users."""

    def __init__(self):
        """Initialize the Keycloak manager."""
        self.admin_client = KeycloakAdmin(
            server_url=app_settings.keycloak_server_url,
            username=app_settings.keycloak_admin_username,
            password=app_settings.keycloak_admin_password,
            realm_name=app_settings.keycloak_realm_name,
            user_realm_name=app_settings.keycloak_realm_name,
            # client_id=app_settings.keycloak_client_id,
            verify=app_settings.keycloak_verify_ssl,
        )

    async def create_realm(self, realm_name: str) -> dict | bytes:
        """Create a new realm in Keycloak."""
        # normalize realm_name
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

        return self.admin_client.create_realm(payload=realm_representation, skip_exists=True)

    async def create_client(self, client_id: str, realm_name: str) -> tuple[str, str]:
        """Create a new client in Keycloak."""
        client_representation = {
            "clientId": client_id,
            "enabled": True,
            "protocol": "openid-connect",
            "publicClient": False,
            "redirectUris": [
                f"{'https' if app_settings.env != 'development' else 'http'}://{app_settings.api_root}/*"
            ],  # TODO: Cross check with the env valaues
            "webOrigins": [f"{'https' if app_settings.env != 'development' else 'http'}://{app_settings.api_root}"],
        }

        try:
            new_client_id = self.admin_client.create_client(payload=client_representation)
            client_secret = self.admin_client.get_client_secrets(new_client_id)["value"]
            return new_client_id, client_secret
        except Exception as e:
            logger.error(f"Error creating client {client_id} in realm {realm_name}: {str(e)}")
            raise e

    async def create_keycloak_user(
        self,
        username: str,
        email: str,
        password: str,
        first_name: str,
        last_name: str,
    ):
        """Create a super user in Keycloak."""
        user = {
            "username": username,
            "email": email,
            "enabled": True,
            "firstName": first_name,
            "lastName": last_name,
            "emailVerified": True,
            "credentials": [{"type": "password", "value": password, "temporary": False}],
        }

        user_id = self.admin_client.create_user(user)
