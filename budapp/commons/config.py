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

"""Manages application and secret configurations, utilizing environment variables and Dapr's configuration store for syncing."""

import os
from pathlib import Path
from typing import Annotated, Any, List, Optional

from budmicroframe.commons.config import (
    BaseAppConfig,
    BaseConfig,
    BaseSecretsConfig,
    enable_periodic_sync_from_store,
    register_settings,
)
from dotenv import load_dotenv
from pydantic import (
    AnyHttpUrl,
    AnyUrl,
    BeforeValidator,
    DirectoryPath,
    Field,
    computed_field,
)

from budapp.__about__ import __version__

from . import logging


load_dotenv()


def parse_cors(v: Any) -> List[str] | str:
    """Parse CORS_ORIGINS into a list of strings."""
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class AppConfig(BaseAppConfig):
    """Manages configuration settings for the microservice.

    This class is used to define and access the configuration settings for the microservice. It supports syncing
    fields from a dapr config store and allows configuration via environment variables.

    Attributes:
        env (str): The environment in which the application is running (e.g., 'development', 'production').
        debug (Optional[bool]): Enable or disable debugging mode.
        Other mandatory fields as required by the application.

    Sync Details:
        Fields annotated with `json_schema_extra` will be synced from the config store. The sync behavior is controlled
        by `sync=True` and additional configurations can be made using `key_prefix` and `alias`.

    Usage:
        Configure the settings via environment variables or sync with a config store. Access settings as attributes of
        an instance of `AppConfig`.

    Example:
        ```python
        from budapp.commons.config import app_settings

        if app_settings.env == "dev":
            # Development-specific logic
            ...
        ```
    """

    # App Info
    name: str = __version__.split("@")[0]
    version: str = __version__.split("@")[-1]
    description: str = ""
    api_root: str = ""

    # Base Directory
    base_dir: DirectoryPath = Field(default_factory=lambda: Path(__file__).parent.parent.parent.resolve())

    # Profiling
    profiler_enabled: bool = Field(False, alias="ENABLE_PROFILER")

    # DB connection
    # postgres_host: str = Field("localhost", alias="POSTGRES_HOST")
    # postgres_port: int = Field(5432, alias="POSTGRES_PORT")
    # postgres_user: str = Field(alias="POSTGRES_USER")
    # postgres_password: str = Field(alias="POSTGRES_PASSWORD")
    # postgres_db: str = Field(alias="POSTGRES_DB")
    postgres_url: Optional[str] = None

    # Superuser
    superuser_email: str = Field(alias="SUPER_USER_EMAIL")
    superuser_password: str = Field(alias="SUPER_USER_PASSWORD")

    # default non master realm name
    default_realm_name: str = Field(alias="DEFAULT_REALM_NAME", default="bud")

    # Token
    access_token_expire_minutes: int = Field(30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    refresh_token_expire_minutes: int = Field(60 * 24 * 7, alias="REFRESH_TOKEN_EXPIRE_MINUTES")

    # Static
    static_dir_path: DirectoryPath | None = Field(None, alias="STATIC_DIR")

    # CORS
    cors_origins: Annotated[list[AnyUrl] | str, BeforeValidator(parse_cors)] = []

    # Bud microservice
    dapr_base_url: AnyHttpUrl = Field(alias="DAPR_BASE_URL")
    bud_cluster_app_id: str = Field(alias="BUD_CLUSTER_APP_ID")
    bud_model_app_id: str = Field(alias="BUD_MODEL_APP_ID")
    bud_simulator_app_id: str = Field(alias="BUD_SIMULATOR_APP_ID")
    bud_metrics_app_id: str = Field(alias="BUD_METRICS_APP_ID")
    bud_notify_app_id: str = Field(alias="BUD_NOTIFY_APP_ID")
    source_topic: str = Field(alias="SOURCE_TOPIC", default="budAppMessages")

    # Budserve host
    budserve_host: str = Field(alias="BUD_SERVE_HOST", default="https://api-dev.bud.studio")

    # Prometheus URL
    prometheus_url: str = Field(alias="PROMETHEUS_URL", default="https://metrics.fmops.in")

    # Add model directory
    add_model_dir: DirectoryPath = Field(os.path.expanduser("~/.cache"), alias="ADD_MODEL_DIR")

    # Keycloak
    keycloak_server_url: str = Field(alias="KEYCLOAK_SERVER_URL")
    keycloak_admin_username: str = Field(alias="KEYCLOAK_ADMIN_USERNAME")
    keycloak_admin_password: str = Field(alias="KEYCLOAK_ADMIN_PASSWORD")
    keycloak_realm_name: str = Field(alias="KEYCLOAK_REALM_NAME")
    keycloak_verify_ssl: bool = Field(True, alias="KEYCLOAK_VERIFY_SSL")

    @computed_field
    def static_dir(self) -> str:
        """Get the static directory."""
        if self.static_dir_path is None:
            return os.path.join(str(self.base_dir), "static")

        return self.static_dir_path

    @computed_field
    def icon_dir(self) -> DirectoryPath:
        """Directory for icon."""
        return os.path.join(self.static_dir, "icons")


# class SecretsConfig(BaseSecretsConfig):
class SecretsConfig(BaseConfig):
    """Manages secret configurations for the microservice.

    This class handles the configuration of secrets required by the microservice. It supports secret management via
    a Dapr secret store.

    Attributes:
        dapr_api_token (str): The API token required for Dapr interactions (mandatory field).
        Other placeholder fields for secrets which can be removed or customized as needed.

    Configuration:
        Secrets should be defined in a `.env` file located in the project root. Use the prefix `SECRETS_` for syncing them with the secret store.
        For example:

        ```
        DAPR_API_TOKEN = your_api_token_here  # Won't be synced to the secret store
        SECRETS_OPENAI_TOKEN = your_api_token_here  # Will be synced to the secret store
        ```

    Sync Details:
        Similar to `AppConfig`, fields can be synced from the secret store using the `json_schema` settings.

    Usage:
        Configure secrets in `.env` for development. Ensure that the `.env` file is not included in the repository.

    Example:
        ```python
        from budapp.commons.config import secrets_settings

        api_token = secrets_settings.dapr_api_token
        ```
    """

    # App Info
    name: str = __version__.split("@")[0]
    version: str = __version__.split("@")[-1]

    dapr_api_token: Optional[str] = Field(None, alias="DAPR_API_TOKEN")

    password_salt: str = Field("bud_password_salt", alias="PASSWORD_SALT")
    jwt_secret_key: str = Field(alias="JWT_SECRET_KEY")
    redis_password: Optional[str] = Field(
        None, alias="REDIS_PASSWORD", json_schema_extra=enable_periodic_sync_from_store(is_global=True)
    )
    redis_uri: Optional[str] = Field(
        None, alias="REDIS_URI", json_schema_extra=enable_periodic_sync_from_store(is_global=True)
    )
    # postgres_user: Optional[str] = Field(
    #     None,
    #     alias="POSTGRES_USER",
    #     json_schema_extra=enable_periodic_sync_from_store(is_global=True),
    # )
    # postgres_password: Optional[str] = Field(
    #     None,
    #     alias="POSTGRES_PASSWORD",
    #     json_schema_extra=enable_periodic_sync_from_store(is_global=True),
    # )
    psql_user: Optional[str] = Field(
        None,
        alias="POSTGRES_USER",
        json_schema_extra=enable_periodic_sync_from_store(is_global=True),
    )
    psql_password: Optional[str] = Field(
        None,
        alias="POSTGRES_PASSWORD",
        json_schema_extra=enable_periodic_sync_from_store(is_global=True),
    )
    hf_token: Optional[str] = Field(
        None, alias="HF_TOKEN", json_schema_extra=enable_periodic_sync_from_store(is_global=True)
    )

    @computed_field
    def redis_url(self) -> str:
        """Construct and returns a Redis connection URL."""
        return f"redis://:{self.redis_password}@{self.redis_uri}"


app_settings = AppConfig()
secrets_settings = SecretsConfig()


def postgres_url(app_settings: BaseAppConfig, secrets_settings: BaseSecretsConfig) -> str:
    """Construct and returns a PostgreSQL connection URL.

    This property combines the individual PostgreSQL connection parameters
    into a single connection URL string.

    Returns:
        A formatted PostgreSQL connection string.
    """
    return f"postgresql://{secrets_settings.psql_user}:{secrets_settings.psql_password}@{app_settings.psql_host}:{app_settings.psql_port}/{app_settings.psql_dbname}"


logging.configure_logging(app_settings.log_dir, app_settings.log_level)


app_settings.postgres_url = postgres_url(app_settings=app_settings, secrets_settings=secrets_settings)

# secrets_settings.psql_user = secrets_settings.postgres_user
# secrets_settings.psql_password = secrets_settings.postgres_password

register_settings(app_settings, secrets_settings)
