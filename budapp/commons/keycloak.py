# budapp/commons/keycloak.py

import base64
import json
import sys
from typing import Dict, List, Optional, Tuple

import requests
from jose import ExpiredSignatureError, JWTError
from keycloak import (
    KeycloakAdmin,
    KeycloakAuthenticationError,
    KeycloakGetError,
    KeycloakInvalidTokenError,
    KeycloakOpenID,
)

from budapp.auth.schemas import ResourceCreate, UserCreate
from budapp.commons.constants import UserRoleEnum
from budapp.user_ops.schemas import TenantClientSchema

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

    async def update_user_password(self, user_id: str, password: str, realm_name: str) -> None:
        """Update a user's password in Keycloak.

        Args:
            user_id: The ID of the user to update
            password: The new password for the user

        Returns:
            None
        """
        try:
            realm_admin = self.get_realm_admin(realm_name)
            realm_admin.set_user_password(user_id=user_id, password=password, temporary=False)
        except Exception as e:
            logger.error(f"Error updating user password: {str(e)}")
            raise

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
                relam_admin.create_realm_role(
                    {"name": role_name, "description": f" Role: {role_name}"}, skip_exists=True
                )

            # Fetch Relm Info
            realm_info = self.admin_client.get_realm(realm_name)
            logger.info(f"Realm {realm_name} created successfully with info: {realm_info}")

            return realm_info

        except Exception as e:
            logger.error(f"Failed to create realm {realm_name}: {str(e)}")
            raise

    async def _create_module_resource(self, realm_admin, client_id: str, module_name: str, realm_name: str) -> str:
        """Create a module-level resource in Keycloak Authorization.

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
            "scopes": [{"name": "view"}, {"name": "manage"}],
        }

        resource_data = None
        existing_resources = realm_admin.get_client_authz_resources(client_id)
        for r in existing_resources:
            if r["name"] == resource["name"]:
                resource_data = r["_id"]
                break

        if not resource_data:
            resource_data = realm_admin.create_client_authz_resource(client_id, resource)

        # Create Permission For The Resource
        kc_scopes = realm_admin.get_client_authz_scopes(client_id)
        logger.debug(f"KC Scopes: {kc_scopes}")
        for scope in kc_scopes:
            permission = {
                "name": f"urn:bud:permission:{module_name}:module:{scope['name']}",
                "description": f"Permission for {module_name} to {scope['name']}",
                "resources": [resource_data["_id"]],
                "policies": [],
                "type": "scope",
                "logic": "POSITIVE",
                "scopes": [scope["id"]],
                "decisionStrategy": "UNANIMOUS",
            }
            logger.debug(f"Creating permission: {json.dumps(permission, indent=4)}")
            permission_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/permission/scope"
            data_raw = realm_admin.connection.raw_post(
                permission_url,
                data=json.dumps(permission),
                max=-1,
                permission=False,
            )

            logger.debug(f"Permission response: {data_raw.json()}")
            permission_id = data_raw.json()["id"]

            logger.debug(f"Permission ID: {permission_id}")

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
            client_secret = secrets.get("value", "")  # type: ignore
            if not client_secret:
                raise ValueError(f"Client secret not found for client {new_client_id}")

            # Create Resources / Scopes / Policies / Roles
            modules = [
                "cluster",
                "model",
                "project",
                "user",
            ]
            for module in modules:
                await self._create_module_resource(realm_admin, new_client_id, module, realm_name)

            return new_client_id, client_secret
        except Exception as e:
            logger.error(f"Error creating client {client_id} in realm {realm_name}: {str(e)}")
            raise e

    async def create_user(self, username: str, email: str, password: str, realm_name: str, role: UserRoleEnum) -> str:
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
            "firstName": user.name,
            "lastName": user.name,
            "enabled": True,
            "emailVerified": True,
            "credentials": [{"type": "password", "value": user.password, "temporary": False}],
        }

        try:
            realm_admin = self.get_realm_admin(realm_name)
            user_id = realm_admin.create_user(payload=user_representation)
            logger.info(f"User {(user.email,)} created successfully")
            logger.debug(f"User ID of realm admin {user.email}: {user_id}")

            roles = realm_admin.get_realm_roles()
            admin_role = next((r for r in roles if r["name"] == user.role.value), None)

            if not admin_role:
                raise ValueError(f"Realm role {user.role.value} not found in realm {realm_name}")

            # Step 2: Assign the role to the user
            realm_admin.assign_realm_roles(user_id=user_id, roles=[admin_role])

            logger.debug(f"Role {user.role.value} assigned to user {user.email}")

            user_permission_map = {}
            if user.permissions:
                for permission in user.permissions:
                    key = permission.name.split(":")[0]
                    scope_name = permission.name.split(":")[1]
                    if key not in user_permission_map:
                        user_permission_map[key] = []
                    user_permission_map[key].append(scope_name)

            # User Policy Name
            policy_name = f"urn:bud:policy:{user_id}"

            for module, scopes in user_permission_map.items():
                logger.debug(f"Module Name: {module}, Scopes: {scopes}")

                resource_name = f"module_{module}"
                resources = realm_admin.get_client_authz_resources(client_id)
                module_resource = next((r for r in resources if r["name"] == resource_name), None)

                if not module_resource:
                    logger.warning(f"Resource {resource_name} not found for client {client_id}")
                    continue

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
                    user_policy_id = data_raw.json()["id"]
                else:
                    user_policy_id = user_policy["id"]

                # Get the permission associated for the resource
                for scope_name in scopes:
                    all_permissions = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/permission?name=urn%3Abud%3Apermission%3A{module}%3Amodule%3A{scope_name}&scope={scope_name}&type=scope"
                    data_raw = realm_admin.connection.raw_get(
                        all_permissions,
                        max=-1,
                        permission=False,
                    )

                    logger.debug(f"All permissions response: {data_raw.json()}")
                    permission_id = data_raw.json()[0]["id"]

                    logger.debug(f"Permission ID: {permission_id}")

                    # Get The Permission
                    permission_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/permission/scope/{permission_id}"
                    permission_data_raw = realm_admin.connection.raw_get(
                        permission_url,
                        max=-1,
                        permission=False,
                    )

                    logger.debug(f"Permission response: {permission_data_raw.json()}")

                    permission_resources_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/policy/{permission_id}/resources"
                    permission_resources_data_raw = realm_admin.connection.raw_get(
                        permission_resources_url,
                        max=-1,
                        permission=False,
                    )

                    logger.debug(f"Permission resources response: {permission_resources_data_raw.json()}")

                    # get the scopes
                    permission_scopes_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/policy/{permission_id}/scopes"
                    permission_scopes_data_raw = realm_admin.connection.raw_get(
                        permission_scopes_url,
                        max=-1,
                        permission=False,
                    )

                    logger.debug(f"Permission scopes response: {permission_scopes_data_raw.json()}")

                    permission_policies_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/policy/{permission_id}/associatedPolicies"
                    permission_policies_data_raw = realm_admin.connection.raw_get(
                        permission_policies_url,
                        max=-1,
                        permission=False,
                    )

                    logger.debug(f"Permission policies response: {permission_policies_data_raw.json()}")

                    update_policy = []
                    for policy in permission_policies_data_raw.json():
                        if policy["name"] == policy_name:
                            update_policy.append(policy["id"])

                    # add policy id to the update policy if not already in the list
                    if user_policy_id not in update_policy:
                        update_policy.append(user_policy_id)

                    # Update
                    permission_update_payload = {
                        "id": permission_data_raw.json()["id"],
                        "name": permission_data_raw.json()["name"],
                        "description": permission_data_raw.json()["description"],
                        "type": permission_data_raw.json()["type"],
                        "logic": permission_data_raw.json()["logic"],
                        "decisionStrategy": permission_data_raw.json()["decisionStrategy"],
                        "resources": [permission_resources_data_raw.json()[0]["_id"]],
                        "policies": update_policy,
                        "scopes": [permission_scopes_data_raw.json()[0]["id"]],
                    }

                    logger.debug(f"Permission update payload: {json.dumps(permission_update_payload, indent=4)}")

                    # Update the permission
                    permission_update_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/permission/scope/{permission_id}"
                    realm_admin.connection.raw_put(
                        permission_update_url,
                        data=json.dumps(permission_update_payload),
                        max=-1,
                        permission=False,
                    )

            return user_id
        except Exception as e:
            logger.error(f"Error creating realm admin {(user.email,)} in realm {realm_name}: {str(e)}", exc_info=True)
            raise

    async def create_realm_admin(
        self, username: str, email: str, password: str, realm_name: str, client_id: str, client_secret: str
    ) -> str:
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
            "firstName": "Admin",  # TODO Update this to be dynamic
            "lastName": "Admin",  # TODO Update this to be dynamic
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
            modules = [
                "cluster",
                "model",
                "project",
                "user",
            ]
            for module in modules:
                resource_name = f"module_{module}"
                resources = realm_admin.get_client_authz_resources(client_id=client_id)
                module_resource = next((r for r in resources if r["name"] == resource_name), None)

                if not module_resource:
                    logger.warning(f"Resource {resource_name} not found for client {client_id}")
                    continue

                resource_id = module_resource["_id"]
                policy_name = f"urn:bud:policy:{user_id}"
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
                    user_policy_id = data_raw.json()["id"]
                else:
                    user_policy_id = user_policy["id"]

                # Get the permission associated for the resource
                scopes = ["view", "manage"]
                for scope in scopes:
                    all_permissions = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/permission?name=urn%3Abud%3Apermission%3A{module}%3Amodule%3A{scope}&scope={scope}&type=scope"
                    data_raw = realm_admin.connection.raw_get(
                        all_permissions,
                        max=-1,
                        permission=False,
                    )

                    logger.debug(f"All permissions response: {data_raw.json()}")
                    permission_id = data_raw.json()[0]["id"]

                    logger.debug(f"Permission ID: {permission_id}")

                    # Get The Permission
                    permission_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/permission/scope/{permission_id}"
                    permission_data_raw = realm_admin.connection.raw_get(
                        permission_url,
                        max=-1,
                        permission=False,
                    )

                    logger.debug(f"Permission response: {permission_data_raw.json()}")

                    # {
                    #     "id": "3697f5d3-5104-4172-a716-cb20e7567383",
                    #     "name": "urn:bud:permission:cluster:manage",
                    #     "description": "Permission for cluster to manage",
                    #     "type": "scope",
                    #     "logic": "POSITIVE",
                    #     "decisionStrategy": "UNANIMOUS"
                    # }

                    # Get the assosciated resource
                    permission_resources_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/policy/{permission_id}/resources"
                    permission_resources_data_raw = realm_admin.connection.raw_get(
                        permission_resources_url,
                        max=-1,
                        permission=False,
                    )

                    logger.debug(f"Permission resources response: {permission_resources_data_raw.json()}")

                    # [{"name":"module_cluster","_id":"b6be1355-d4d6-4271-aa71-1f6986b11419"}]

                    # get the scopes
                    permission_scopes_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/policy/{permission_id}/scopes"
                    permission_scopes_data_raw = realm_admin.connection.raw_get(
                        permission_scopes_url,
                        max=-1,
                        permission=False,
                    )

                    logger.debug(f"Permission scopes response: {permission_scopes_data_raw.json()}")

                    # [{"id":"370c6c54-1471-4728-97d7-fd8d97b6b4c2","name":"manage"}]

                    # get associated policies and check if our policy is one of them
                    permission_policies_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/policy/{permission_id}/associatedPolicies"
                    permission_policies_data_raw = realm_admin.connection.raw_get(
                        permission_policies_url,
                        max=-1,
                        permission=False,
                    )

                    logger.debug(f"Permission policies response: {permission_policies_data_raw.json()}")

                    # Check if the user policy is one of the associated policies
                    # user_policy_id = next((p["id"] for p in permission_policies_data_raw.json() if p["name"] == policy_name), None)

                    # if not user_policy_id:
                    #     logger.warning(f"User policy {policy_name} not found in associated policies")
                    #     continue

                    update_policy = []
                    for policy in permission_policies_data_raw.json():
                        if policy["name"] == policy_name:
                            update_policy.append(policy["id"])

                    # add policy id to the update policy if not already in the list
                    if user_policy_id not in update_policy:
                        update_policy.append(user_policy_id)

                    #                     [
                    #     {
                    #         "id": "41149b8f-dac0-4fc2-8fa6-02d18f6c71b0",
                    #         "name": "urn:bud:policy:b04b78dd-d6a1-4f05-a9e4-85e10f9e6c58",
                    #         "description": "User policy for b04b78dd-d6a1-4f05-a9e4-85e10f9e6c58",
                    #         "type": "user",
                    #         "logic": "POSITIVE",
                    #         "decisionStrategy": "UNANIMOUS",
                    #         "config": {}
                    #     }
                    # ]

                    # Update the permission
                    # {
                    #     "id": "3697f5d3-5104-4172-a716-cb20e7567383",
                    #     "name": "urn:bud:permission:cluster:manage",
                    #     "description": "Permission for cluster to manage",
                    #     "type": "scope",
                    #     "logic": "POSITIVE",
                    #     "decisionStrategy": "UNANIMOUS",
                    #     "resources": [
                    #         "b6be1355-d4d6-4271-aa71-1f6986b11419"
                    #     ],
                    #     "policies": [
                    #         "41149b8f-dac0-4fc2-8fa6-02d18f6c71b0"
                    #     ],
                    #     "scopes": [
                    #         "370c6c54-1471-4728-97d7-fd8d97b6b4c2"
                    #     ]
                    # }

                    permission_update_payload = {
                        "id": permission_data_raw.json()["id"],
                        "name": permission_data_raw.json()["name"],
                        "description": permission_data_raw.json()["description"],
                        "type": permission_data_raw.json()["type"],
                        "logic": permission_data_raw.json()["logic"],
                        "decisionStrategy": permission_data_raw.json()["decisionStrategy"],
                        "resources": [permission_resources_data_raw.json()[0]["_id"]],
                        "policies": update_policy,
                        "scopes": [permission_scopes_data_raw.json()[0]["id"]],
                    }

                    logger.debug(f"Permission update payload: {json.dumps(permission_update_payload, indent=4)}")

                    # Update the permission
                    permission_update_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/permission/scope/{permission_id}"
                    realm_admin.connection.raw_put(
                        permission_update_url,
                        data=json.dumps(permission_update_payload),
                        max=-1,
                        permission=False,
                    )

                    # logger.debug(f"Permission update response: {permission_update_data_raw.json()}")

                    # logger.info(f"Permission {permission_data_raw.json()['name']} assigned to user {username} for module {module}")

                # Step 3b: Create permission (if not exists)
                # permission_name = f"user-{user_id}-module-{module}"
                # existing_permissions = realm_admin.get_client_authz_permissions(client_id)
                # if any(p["name"] == permission_name for p in existing_permissions):
                #     logger.info(f"Permission {permission_name} already exists — skipping.")
                #     continue

                # # Scope from keycloak
                # kc_scopes = realm_admin.get_client_authz_scopes(client_id)
                # logger.debug(f"KC Scopes: {kc_scopes}")

                # permission = {
                #     "name": permission_name,
                #     "decisionStrategy": "UNANIMOUS",
                #     "description": f"Permission for {user_id} to view and manage {module}",
                #     "resources": [resource_id],
                #     "policies": [policy_id],
                #     "scopes": [scope["id"] for scope in kc_scopes],
                #     "type": "scope",
                #     "logic": "POSITIVE",
                # }

                # logger.debug(f"Creating permission: {json.dumps(permission, indent=4)}")
                # permission_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/permission/scope"
                # data_raw = realm_admin.connection.raw_post(
                #     permission_url,
                #     data=json.dumps(permission),
                #     max=-1,
                #     permission=False,
                # )

                # logger.debug(f"Permission response: {data_raw.json()}")
                # permission_id = data_raw.json()["id"]

                # logger.debug(f"Permission ID: {permission_id}")

                # # realm_admin.create_client_authz_permission(client_id, permission)
                # logger.info(f"Permission {permission_name} assigned to user {username} for module {module}")

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

    async def refresh_token(self, realm_name: str, credentials: TenantClientSchema, refresh_token: str) -> dict:
        """Refresh the access token using the refresh token.

        Args:
            realm_name: Name of the Keycloak realm
            credentials: TenantClientSchema with client_id and client_secret
            refresh_token: The refresh token string

        Returns:
            dict: New token response from Keycloak
        """
        try:
            openid_client = self.get_keycloak_openid_client(realm_name, credentials)
            new_token = openid_client.refresh_token(refresh_token)
            return new_token
        except KeycloakAuthenticationError as e:
            logger.warning(f"Failed to refresh token due to invalid credentials or expired token: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while refreshing token: {str(e)}")
            raise

    async def authenticate_user(
        self, username: str, password: str, realm_name: str, credentials: TenantClientSchema
    ) -> dict:
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

    async def get_user_roles_and_permissions(
        self,
        user_id: str,
        realm_name: str,
        client_id: str,
        credentials: TenantClientSchema,
        token: str,
    ) -> Dict[str, List[str]]:
        try:
            keycloak_openid = self.get_keycloak_openid_client(realm_name, credentials)

            # Validate the endpoint is correct
            oidc_config = keycloak_openid.well_known()
            token_endpoint = oidc_config.get("token_endpoint")
            if not token_endpoint:
                raise Exception("Could not retrieve token_endpoint from OIDC well-known configuration.")

            # access token = token
            if not token:
                raise KeycloakAuthenticationError("Access token not found.")

            # Request the RPT
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/x-www-form-urlencoded",
            }
            payload = {
                "grant_type": "urn:ietf:params:oauth:grant-type:uma-ticket",
                "audience": credentials.client_named_id,
            }

            response = requests.post(token_endpoint, headers=headers, data=payload)
            response.raise_for_status()

            # RPT Response
            rpt_response_data = response.json()
            requesting_party_token = rpt_response_data.get("access_token")

            if not requesting_party_token:
                raise ValueError(
                    f"'access_token' (RPT) not found in UMA response. Full response: {json.dumps(rpt_response_data)}"
                )

            # Decode RPT Payload
            rpt_payload = self._decode_jwt_payload(requesting_party_token)
            if not rpt_payload:
                raise ValueError("Failed to decode the obtained RPT.")

            # Extract and format permissions
            raw_permissions = rpt_payload.get("authorization", {}).get("permissions", [])

            formatted_permissions = []
            for perm in raw_permissions:
                # Ensure we only include the desired fields and handle missing scopes
                formatted_perm = {
                    "rsid": perm.get("rsid"),  # Get rsid or None if missing
                    "rsname": perm.get("rsname"),  # Get rsname or None if missing
                    "scopes": perm.get("scopes", []),  # Get scopes or default to empty list
                }
                formatted_permissions.append(formatted_perm)

            # Prepare final output structure
            final_output = {"permissions": formatted_permissions}

            return final_output

        except Exception as e:
            logger.error(f"Error getting user roles and permissions: {str(e)}")
            raise

    def _decode_jwt_payload(self, token: str) -> dict:
        """Safely decodes the payload of a JWT using base64."""
        try:
            # Split token into parts: header, payload, signature
            parts = token.split(".")
            if len(parts) != 3:
                print("[ERROR] Invalid JWT format: Incorrect number of segments.", file=sys.stderr)
                return None

            payload_encoded = parts[1]
            # Add padding if necessary for base64 decoding
            payload_encoded += "=" * (-len(payload_encoded) % 4)
            # Use urlsafe_b64decode for JWT compatibility
            payload_decoded_bytes = base64.urlsafe_b64decode(payload_encoded)
            payload_json = json.loads(payload_decoded_bytes.decode("utf-8"))
            return payload_json
        except IndexError:
            # This case should be caught by len(parts) check now, but keep for safety
            print("[ERROR] Invalid JWT format: Could not split segments.", file=sys.stderr)
        except (TypeError, base64.binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"[ERROR] Failed to decode JWT payload: {e}", file=sys.stderr)
            print(f"Encoded payload segment was: {payload_encoded}", file=sys.stderr)
        return None

    async def remove_resource_with_permissions(
        self, realm_name: str, client_id: str, resource: ResourceCreate, user_auth_id: str
    ) -> None:
        """Remove a resource with permissions."""
        try:
            if resource.scopes is None:
                logger.info(f"Resource {resource.resource_type} does not have any scopes — skipping.")
                return

            realm_admin = self.get_realm_admin(realm_name)

            # for scope in resource.scopes:
            # logger.info(f"Removing resource {resource.resource_type} with scope {scope} — skipping.")

            # resource_name = f"URN::{resource.resource_type}::{resource.resource_id}::{scope}"

            # resources = realm_admin.get_client_authz_resources(client_id)
            # module_resource = next((r for r in resources if r["name"] == resource_name), None)

            # if not module_resource:
            #     logger.info(f"Resource {resource_name} does not exist — skipping.")
            #     continue

            # resource_id = module_resource["_id"]

            # permission_url = f"{app_settings.keycloak_server_url}/admin/realms/{realm_name}/clients/{client_id}/authz/resource-server/permission/resource"

            # permissions = realm_admin.get_client_authz_permissions(client_id)
            # permission = next((p for p in permissions if p["resource"] == resource_id), None)

            # if not permission:
            #     logger.info(f"Permission for resource {resource_name} does not exist — skipping.")
            #     continue

            # realm_admin.delete_client_authz_permission(client_id, permission["id"])
            # logger.info(f"Permission for resource {resource_name} deleted successfully")

        except Exception as e:
            logger.error(f"Error getting realm admin: {str(e)}")
            raise

    async def create_resource_with_permissions(
        self, realm_name: str, client_id: str, resource: ResourceCreate, user_auth_id: str
    ) -> None:
        """Create a resource with permissions."""
        try:
            realm_admin = self.get_realm_admin(realm_name)

            scopes = ["view", "manage"]

            for scope in scopes:
                resource_name = f"URN::{resource.resource_type}::{resource.resource_id}::{scope}"

                # Check if resource already exists
                resources = realm_admin.get_client_authz_resources(client_id)
                module_resource = next((r for r in resources if r["name"] == resource_name), None)

                if not module_resource:
                    resource = {
                        "name": f"{resource_name}",
                        "type": resource.resource_type.capitalize(),
                        "owner": {"id": client_id},
                        "ownerManagedAccess": True,
                        "displayName": f"{resource_name.capitalize()} Entity",
                        "scopes": [
                            {
                                "name": resource.scopes[0],
                            },
                            {
                                "name": resource.scopes[1],
                            },
                        ],
                    }

                    realm_admin.create_client_authz_resource(client_id, resource)
                    logger.info(f"Resource {resource_name} created successfully")

                # Continue only if the resource has a scope equal to "scope"
                if scope not in resource.scopes:
                    logger.info(f"Resource {resource_name} does not have a scope equal to {scope} — skipping.")
                    continue

                resource_id = module_resource["_id"]
                policy_name = f"user-policy-{user_auth_id}-{resource_name}"
                existing_policies = realm_admin.get_client_authz_policies(client_id)
                logger.debug(f"Existing policies: {existing_policies}")
                user_policy = next((p for p in existing_policies if p["name"] == policy_name), None)

                if not user_policy:
                    user_policy = {
                        "name": policy_name,
                        "description": f"User policy for {user_auth_id}",
                        "logic": "POSITIVE",
                        "users": [user_auth_id],
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

                # Create permission (if not exists)
                permission_name = f"user-{user_auth_id}-entity-{resource.resource_type}-{resource.resource_id}-{scope}"
                existing_permissions = realm_admin.get_client_authz_permissions(client_id)
                if any(p["name"] == permission_name for p in existing_permissions):
                    logger.info(f"Permission {permission_name} already exists — skipping.")
                    return

                permission = {
                    "name": permission_name,
                    "decisionStrategy": "UNANIMOUS",
                    "description": f"Permission for {user_auth_id} to view and manage {resource.resource_type}",
                    "resources": [resource_id],
                    "policies": [policy_id],
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

                logger.info(
                    f"Permission {permission_name} assigned to user {user_auth_id} for {resource.resource_type}"
                )
        except Exception as e:
            logger.error(f"Error creating resource with permissions: {str(e)}")
            raise

    async def validate_token(self, token: str, realm_name: str, credentials: TenantClientSchema) -> dict:
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
