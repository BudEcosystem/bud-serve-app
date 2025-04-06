# budapp/commons/keycloak.py

import json
from typing import Optional, Tuple
from venv import create

from keycloak import KeycloakAdmin, KeycloakAuthenticationError, KeycloakGetError, KeycloakInvalidTokenError, KeycloakOpenID

from budapp.auth.schemas import UserCreate
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
            client_id=str(credentials.client_named_id),
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
            roles = [
                UserRoleEnum.ADMIN.value,
                UserRoleEnum.DEVELOPER.value,
                UserRoleEnum.TESTER.value,
                UserRoleEnum.DEVOPS.value,
                UserRoleEnum.SUPER_ADMIN.value,
            ]
            
            for role_name in roles:
                relam_admin.create_realm_role({"name": role_name, "description": f" Role: {role_name}"}, skip_exists=True)

            # Fetch Relm Info
            realm_info =  self.admin_client.get_realm(realm_name)
            logger.info(f"Realm {realm_name} created successfully with info: {realm_info}")

            return realm_info


        except Exception as e:
            logger.error(f"Failed to create realm {realm_name}: {str(e)}")
            raise
    
    
    async def _create_module_resource(self, realm_admin, client_id: str, module_name: str) -> str:
        """
        Create a module-level resource in Keycloak Authorization.

        Args:
            realm_admin: Instance of realm_admin
            client_id: Internal client UUID
            module_name: e.g. "cluster", "model"

        Returns:
            ID of the created resource
        """
        
        resource = {
            "name": f"module_{module_name}",
            "type": "module",
            "owner": {"id": client_id},
            "ownerManagedAccess": True,
            "displayName": f"{module_name.capitalize()} Module",
            "scopes": [{"name": module_name.split(":")[1]}]
        }
        
        existing_resources = realm_admin.get_client_authz_resources(client_id)
        for r in existing_resources:
            if r["name"] == resource["name"]:
                return r["_id"]
            
        return realm_admin.create_client_authz_resource(client_id, resource)

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
            "redirectUris": ["*"],
            "webOrigins": ["*"],
            "authorizationServicesEnabled": True,
            "serviceAccountsEnabled": True,
        }

        try:
            realm_admin = self.get_realm_admin(realm_name)
            new_client_id = realm_admin.create_client(payload=client_representation)
            secrets = realm_admin.get_client_secrets(new_client_id)
            client_secret = secrets.get("value", "") # type: ignore
            if not client_secret:
                raise ValueError(f"Client secret not found for client {new_client_id}")
            
            # Create Resources / Scopes / Policies / Roles
            modules = ["cluster:view", "model:view", "projects:view", "user:view","cluster:manage", "model:manage", "projects:manage", "user:manage"]
            for module in modules:
                await self._create_module_resource(realm_admin, new_client_id, module)
        
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
        
    async def create_user_with_permissions(self, user: UserCreate, realm_name: str, client_id: str) -> str:
        """Create a new user with permissions in Keycloak.

        Args:
            user: UserCreate object
            realm_name: Name of the realm to create the user in


        Returns:
            str: User ID of the created user
        """
        
        user_representation = {
            "username": user.email,
            "email": user.email,
            "firstName":user.name,
            "lastName":user.name,
            "enabled": True,
            "emailVerified": True,
            "credentials": [{"type": "password", "value": user.password, "temporary": False}],
        }
        
        try:
            realm_admin = self.get_realm_admin(realm_name)
            user_id = realm_admin.create_user(payload=user_representation)
            logger.info(f"User {user.email,} created successfully")
            logger.debug(f"User ID of realm admin {user.email,}: {user_id}")
            
            # assign role to user
            realm_admin.assign_realm_roles(user_id=user_id, roles=[user.role.value])
            logger.info(f"Role {user.role.value} assigned to user {user.email}")
            
            # Assign policies to the user
            if user.permissions:
                for permission in user.permissions:
                    module = permission.name.split(":")[0]
                    permission_base_name = permission.name.split(":")[1]
                    has_permission = permission.has_permission
                    
                    resource_name = f"module_{module}"
                    resources = realm_admin.get_client_authz_resources(client_id)
                    module_resource = next((r for r in resources if r["name"] == resource_name), None)
                    
                    if not module_resource:
                        logger.warning(f"Resource {resource_name} not found for client {client_id}")
                        continue
            
                    resource_id = module_resource["_id"]
                    policy_name = f"user-policy-{user_id}"
                    existing_policies = realm_admin.get_client_authz_policies(client_id)
                    logger.debug(f"Existing policies: {existing_policies}")
                    user_policy = next((p for p in existing_policies if p["name"] == policy_name), None)
                    
                    if not user_policy:
                        user_policy = {
                            "name": policy_name,
                            "description": f"User policy for {user_id}",
                            "logic": "POSITIVE",
                            "users": [user_id],
                        }
                        
                        logger.debug(f"Creating user policy: {json.dumps(user_policy, indent=4)}")
                        
                        policy_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/policy/user"
                        
                        data_raw = realm_admin.connection.raw_post(
                            policy_url, 
                            data=json.dumps(user_policy),
                            max=-1,
                            permission=False,
                        )
                        
                        logger.debug(f"User policy response: {data_raw.json()}")
                        policy_id = data_raw.json()["id"]
                    else:
                        policy_id = user_policy["id"]
                        
                    # Permissions
                    permission_name = f"user-{user_id}-module-{module}-{permission_base_name}"
                    existing_permissions = realm_admin.get_client_authz_permissions(client_id)
                    if any(p["name"] == permission_name for p in existing_permissions):
                        logger.info(f"Permission {permission_name} already exists — skipping.")
                        continue
                    
                    permission = {
                        "name": permission_name,
                        "decisionStrategy": "UNANIMOUS",
                        "description": f"Permission for {user_id} to {permission_base_name} {module}",
                        "resources": [resource_id],
                        "policies": [policy_id]
                    }
                    
                    logger.debug(f"Creating permission: {json.dumps(permission, indent=4)}")
                    
                    permission_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/permission/resource"
                    data_raw = realm_admin.connection.raw_post(
                        permission_url, 
                        data=json.dumps(permission),
                        max=-1,
                        permission=False,
                    )
                    
                    logger.debug(f"Permission response: {data_raw.json()}")
                    permission_id = data_raw.json()["id"]
                    
                    logger.debug(f"Permission ID: {permission_id}")
                    
                
                    logger.info(f"Permission {permission_name} assigned to user {user.email} for module {module}")
                    
            return user_id
        except Exception as e:
            logger.error(f"Error creating realm admin {user.email,} in realm {realm_name}: {str(e)}")
            raise
        
    

    async def create_realm_admin(self, username: str, email: str, password: str, realm_name: str, client_id: str, client_secret: str) -> str:
        """create a new realm admin user in Keycloak.

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
            "firstName":"Admin", #TODO Update this to be dynamic
            "lastName":"Admin", #TODO Update this to be dynamic
            "enabled": True,
            "emailVerified": True,
            "credentials": [{"type": "password", "value": password, "temporary": False}],
        }

        try:
            realm_admin = self.get_realm_admin(realm_name)
            user_id = realm_admin.create_user(payload=user_representation, exist_ok=True)
            logger.info(f"Realm admin {username} created successfully")

            logger.debug(f"User ID of realm admin {username}: {user_id}")
            
            # Step 1: Get all realm roles
            roles = realm_admin.get_realm_roles()
            super_admin_role = next((r for r in roles if r["name"] == "super_admin"), None)

            if not super_admin_role:
                raise ValueError(f"Realm role 'super_admin' not found in realm {realm_name}")
            
            # Step 2: Assign the role to the user
            realm_admin.assign_realm_roles(user_id=user_id, roles=[super_admin_role])
            logger.info(f"Assigned 'super_admin' role to user {username}")
            
            # Step 3: Assign policies to the user
            modules = ["cluster:view", "model:view", "projects:view", "user:view","cluster:manage", "model:manage", "projects:manage", "user:manage"]
            for module in modules:
                resource_name = f"module_{module}"
                resources = realm_admin.get_client_authz_resources(client_id=client_id)
                module_resource = next((r for r in resources if r["name"] == resource_name), None)
                
                if not module_resource:
                    logger.warning(f"Resource {resource_name} not found for client {client_id}")
                    continue
                
                resource_id = module_resource["_id"]
                policy_name = f"user-policy-{user_id}"
                existing_policies = realm_admin.get_client_authz_policies(client_id)
                logger.debug(f"Existing policies: {existing_policies}")
                user_policy = next((p for p in existing_policies if p["name"] == policy_name), None)
                
                if not user_policy:
                    user_policy = {
                        "name": policy_name,
                        "description": f"User policy for {user_id}",
                        "logic": "POSITIVE",
                        "users": [user_id],
                    }
                    
                    logger.debug(f"Creating user policy: {json.dumps(user_policy, indent=4)}")
                    
                    policy_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/policy/user"
                    
                    data_raw = realm_admin.connection.raw_post(
                        policy_url, 
                        data=json.dumps(user_policy),
                        max=-1,
                        permission=False,
                    )
                    
                    logger.debug(f"User policy response: {data_raw.json()}")
                    policy_id = data_raw.json()["id"]
                else:
                    policy_id = user_policy["id"]
                    
                # Step 3b: Create permission (if not exists)
                permission_name = f"user-{user_id}-module-{module}"
                existing_permissions = realm_admin.get_client_authz_permissions(client_id)
                if any(p["name"] == permission_name for p in existing_permissions):
                    logger.info(f"Permission {permission_name} already exists — skipping.")
                    continue

                permission = {
                    "name": permission_name,
                    "decisionStrategy": "UNANIMOUS",
                    "description": f"Permission for {user_id} to view and manage {module}",
                    "resources": [resource_id],
                    "policies": [policy_id]
                }
                
                logger.debug(f"Creating permission: {json.dumps(permission, indent=4)}")
                permission_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/permission/resource"
                data_raw = realm_admin.connection.raw_post(
                    permission_url, 
                    data=json.dumps(permission),
                    max=-1,
                    permission=False,
                )
                
                logger.debug(f"Permission response: {data_raw.json()}")
                permission_id = data_raw.json()["id"]
                
                logger.debug(f"Permission ID: {permission_id}")
                
                #realm_admin.create_client_authz_permission(client_id, permission)
                logger.info(f"Permission {permission_name} assigned to user {username} for module {module}")

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
            token = openid_client.token(username, password, scope="openid profile email roles")
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
