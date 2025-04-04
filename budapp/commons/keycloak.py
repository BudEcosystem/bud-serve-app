# budapp/commons/keycloak.py

from typing import Optional, Tuple

from keycloak import KeycloakAdmin, KeycloakAuthenticationError, KeycloakGetError, KeycloakInvalidTokenError, KeycloakOpenID

from budapp.commons.constants import UserRoleEnum
from budapp.user_ops.schemas import TenantClientSchema
from jose import JWTError, ExpiredSignatureError


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
        if self._admin_client is None:
            raise ValueError("KeycloakAdmin client not initialized")
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
    def get_keycloak_openid_client(self, realm_name: str, credentials: TenantClientSchema) -> KeycloakOpenID:
        """Get the keycloak openid client.

        Args:
            realm_name: Name of the realm
            credentials: The client credentials schema containing client_id and client_secret
        """
        return KeycloakOpenID(
            server_url=app_settings.keycloak_server_url,
            client_id=str(credentials.client_id),
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

            # MAP Roles ad Groups
            relam_admin = self.get_realm_admin(realm_name)
            groups = [
                UserRoleEnum.ADMIN.value,
                UserRoleEnum.DEVELOPER.value,
                UserRoleEnum.TESTER.value,
                UserRoleEnum.DEVOPS.value,
                UserRoleEnum.SUPER_ADMIN.value,
            ]

            for group in groups:
                relam_admin.create_group({"name": group, "path": f"/{group}"}, skip_exists=True)

            # Create permission roles in Keycloak
            role_names = [
                "model:view", "model:manage",
                "cluster:view", "cluster:manage",
                "user:view", "user:manage",
                "projects:view", "projects:manage",
            ]

            for role_name in role_names:
                relam_admin.create_realm_role({"name": role_name, "description": f"Permission role: {role_name}"}, skip_exists=True)

            # Step 1: Get the role objects (dicts) by name
            # role_objs = [
            #     relam_admin.get_realm_role(role_name)
            #     for role_name in role_names
            # ]

            # # Get group objects and assign roles
            # existing_groups = relam_admin.get_groups()
            # for group in existing_groups:
            #     if group["name"] in groups:
            #         relam_admin.assign_group_realm_roles(group_id=group["id"], roles=role_objs)

            # Fetch Relm Info
            realm_info =  self.admin_client.get_realm(realm_name)
            logger.info(f"Realm {realm_name} created successfully with info: {realm_info}")

            return realm_info


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
            secrets = realm_admin.get_client_secrets(new_client_id)
            client_secret = secrets.get("value", "") # type: ignore
            if not client_secret:
                raise ValueError(f"Client secret not found for client {new_client_id}")
            return new_client_id, client_secret
        except Exception as e:
            logger.error(f"Error creating client {client_id} in realm {realm_name}: {str(e)}")
            raise e

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
            realm_admin = self.get_realm_admin(realm_name)
            user_id = realm_admin.create_user(payload=user_representation, exist_ok=True)
            logger.info(f"Realm admin {username} created successfully")

            logger.debug(f"User ID of realm admin {username}: {user_id}")

            groups = realm_admin.get_groups()
            super_admin_group = next((g for g in groups if g["name"] == UserRoleEnum.SUPER_ADMIN.value), None)

            if not super_admin_group:
                raise ValueError(f"Group {UserRoleEnum.SUPER_ADMIN.value} not found in realm {realm_name}")

            realm_admin.group_user_add(user_id=user_id, group_id=super_admin_group["id"])
            logger.info(f"Assigned user {username} to group {UserRoleEnum.SUPER_ADMIN.value}")

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

    async def authenticate_user(self, username: str, password: str, realm_name: str, credentials: TenantClientSchema) -> dict:
        """Authenticate a user and return access & refresh tokens.

        Args:
            username: Username of the user to authenticate
            password: Password of the user to authenticate
            realm_name: Name of the realm to authenticate the user in
            credentials: Contains client_id and (optional) client_secret

        Returns:
            dict: Contains access_token, refresh_token, etc.

        Example:
            {
                "access_token": "eyJhbGciOiJSUzI1NiIsInR...",
                "expires_in": 300,
                "refresh_expires_in": 1800,
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR...",
                "token_type": "Bearer",
                "not-before-policy": 0,
                "session_state": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
                "scope": "email profile"
            }
        """
        try:
            openid_client = self.get_keycloak_openid_client(realm_name, credentials)
            token = openid_client.token(username, password)
            return token
        except KeycloakAuthenticationError:
            logger.warning(f"Invalid credentials for user {username}")
            return {}
        except Exception as e:
            logger.error(f"Error verifying password for user {username}: {str(e)}")
            return {}

    async def logout_user(self, refresh_token: str, realm_name: str, credentials: TenantClientSchema) -> bool:
        """Log out a user by invalidating the refresh token.

        Args:
            refresh_token: The refresh token to invalidate
            realm_name: The realm the user belongs to
            credentials: Contains client_id and (optional) client_secret

        Returns:
            bool: True if logout was successful, False otherwise
        """
        try:
            openid_client = self.get_keycloak_openid_client(realm_name, credentials)
            openid_client.logout(refresh_token)
            logger.info(f"User successfully logged out from realm {realm_name}")
            return True
        except Exception as e:
            logger.error(f"Error logging out user from realm {realm_name}: {str(e)}")
            return False

    async def validate_token(
        self,
        token: str,
        realm_name: str,
        credentials: TenantClientSchema
    ) -> dict:
        """Validate a JWT token and return the decoded claims if valid.

        Args:
            token: JWT access token
            realm_name: Realm name the token belongs to
            credentials: Client info (used to initialize OpenID client)

        Returns:
            dict: Decoded token claims if valid, else raises exception

        Example:
        {
            "exp": 1712180647,
            "iat": 1712180347,
            "auth_time": 1712180345,
            "jti": "abc123-def456",
            "iss": "https://keycloak.example.com/realms/myrealm",
            "aud": "my-client-id",
            "sub": "bf7acff1-2b9c-4936-bde2-3f607f3e3c67",
            "typ": "Bearer",
            "azp": "my-client-id",
            "session_state": "b928e15b-b3ff-4f18-9ff9-f19dd29d5142",
            "acr": "1",
            "realm_access": {
                "roles": [
                "user",
                "admin"
                ]
            },
            "resource_access": {
                "my-client-id": {
                "roles": ["reader", "writer"]
                }
            },
            "scope": "email profile",
            "email_verified": True,
            "preferred_username": "john.doe"
            }
        """
        try:
            openid_client = self.get_keycloak_openid_client(realm_name, credentials)

            decoded_token = openid_client.decode_token(
                token,
            )

            logger.info(f"Token successfully validated for realm {realm_name}")
            return decoded_token

        except (KeycloakAuthenticationError, KeycloakGetError) as e:
            logger.error(f"Keycloak error during token validation: {str(e)}")
            raise

        except ExpiredSignatureError:
            logger.warning("Token has expired")
            raise

        except JWTError as e:
            logger.warning(f"General JWT error: {str(e)}")
            raise
        
        except KeycloakInvalidTokenError as e:
            logger.warning(f"Keycloak invalid token error: {str(e)}")
            raise

        except Exception as e:
            logger.error(f"Unexpected error validating token: {str(e)}")
            raise
