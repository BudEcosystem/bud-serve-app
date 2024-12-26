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

"""Defines constant values used throughout the project, including application-specific constants."""

import random
from enum import Enum, StrEnum, auto
from typing import List

from .helpers import create_dynamic_enum


class LogLevel(Enum):
    """Define logging levels with associated priority values.

    Inherit from `str` and `Enum` to create a logging level enumeration. Each level has a string representation and a
    corresponding priority value, which aligns with Python's built-in `logging` module levels.

    Attributes:
        DEBUG (LogLevel): Represents debug-level logging with a priority value of `logging.DEBUG`.
        INFO (LogLevel): Represents info-level logging with a priority value of `logging.INFO`.
        WARNING (LogLevel): Represents warning-level logging with a priority value of `logging.WARNING`.
        ERROR (LogLevel): Represents error-level logging with a priority value of `logging.ERROR`.
        CRITICAL (LogLevel): Represents critical-level logging with a priority value of `logging.CRITICAL`.
        NOTSET (LogLevel): Represents no logging level with a priority value of `logging.NOTSET`.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    NOTSET = "NOTSET"


class Environment(str, Enum):
    """Enumerate application environments and provide utilities for environment-specific settings.

    Inherit from `str` and `Enum` to define application environments with associated string values. The class also
    includes utility methods to convert string representations to `Environment` values and determine logging and
    debugging settings based on the environment.

    Attributes:
        PRODUCTION (Environment): Represents the production environment.
        DEVELOPMENT (Environment): Represents the development environment.
        TESTING (Environment): Represents the testing environment.
    """

    PRODUCTION = "PRODUCTION"
    DEVELOPMENT = "DEVELOPMENT"
    TESTING = "TESTING"

    @staticmethod
    def from_string(value: str) -> "Environment":
        """Convert a string representation to an `Environment` instance.

        Use regular expressions to match and identify the environment from a string. Raise a `ValueError` if the string
        does not correspond to a valid environment.

        Args:
            value (str): The string representation of the environment.

        Returns:
            Environment: The corresponding `Environment` instance.

        Raises:
            ValueError: If the string does not match any valid environment.
        """
        import re

        matches = re.findall(r"(?i)\b(dev|prod|test)(elop|elopment|uction|ing|er)?\b", value)

        env = matches[0][0].lower() if len(matches) else ""
        if env == "dev":
            return Environment.DEVELOPMENT
        elif env == "prod":
            return Environment.PRODUCTION
        elif env == "test":
            return Environment.TESTING
        else:
            raise ValueError(
                f"Invalid environment: {value}. Only the following environments are allowed: "
                f"{', '.join(map(str, Environment.__members__))}"
            )

    @property
    def log_level(self) -> LogLevel:
        """Return the appropriate logging level for the current environment.

        Returns:
            LogLevel: The logging level for the current environment.
        """
        return {"PRODUCTION": LogLevel.INFO}.get(self.value, LogLevel.DEBUG)

    @property
    def debug(self) -> bool:
        """Return whether debugging is enabled for the current environment.

        Returns:
            bool: `True` if debugging is enabled, `False` otherwise.
        """
        return {"PRODUCTION": False}.get(self.value, True)


class ModalityEnum(Enum):
    """Enumeration of model modalities.

    This enum represents different types of AI model modalities or capabilities.

    Attributes:
        LLM (str): Represents Large Language Models for text generation and processing.
        IMAGE (str): Represents image-related models for tasks like generation or analysis.
        EMBEDDING (str): Represents models that create vector embeddings of input data.
        TEXT_TO_SPEECH (str): Represents models that convert text to spoken audio.
        SPEECH_TO_TEXT (str): Represents models that transcribe spoken audio to text.
    """

    LLM = "llm"
    IMAGE = "image"
    EMBEDDING = "embedding"
    TEXT_TO_SPEECH = "text_to_speech"
    SPEECH_TO_TEXT = "speech_to_text"


ModelSourceEnum = create_dynamic_enum(
    "ModelSourceEnum",
    [
        "local",
        "nlp_cloud",
        "deepinfra",
        "anthropic",
        "vertex_ai-vision-models",
        "vertex_ai-ai21_models",
        "cerebras",
        "watsonx",
        "predibase",
        "volcengine",
        "clarifai",
        "baseten",
        "sambanova",
        "github",
        "petals",
        "replicate",
        "vertex_ai-chat-models",
        "azure_ai",
        "perplexity",
        "vertex_ai-code-text-models",
        "vertex_ai-text-models",
        "cohere_chat",
        "vertex_ai-embedding-models",
        "text-completion-openai",
        "groq",
        "openai",
        "aleph_alpha",
        "sagemaker",
        "databricks",
        "fireworks_ai",
        "vertex_ai-anthropic_models",
        "vertex_ai-mistral_models",
        "voyage",
        "vertex_ai-language-models",
        "anyscale",
        "deepseek",
        "vertex_ai-image-models",
        "mistral",
        "ollama",
        "cohere",
        "gemini",
        "friendliai",
        "vertex_ai-code-chat-models",
        "azure",
        "codestral",
        "vertex_ai-llama_models",
        "together_ai",
        "cloudflare",
        "ai21",
        "openrouter",
        "bedrock",
        "text-completion-codestral",
        "huggingface",
    ],
)

CredentialTypeEnum = Enum(
    "CredentialTypeEnum",
    {
        name: member.value
        for name, member in ModelSourceEnum.__members__.items()
        if member not in [ModelSourceEnum.LOCAL]
    },
)


class ModelProviderTypeEnum(Enum):
    """Enumeration of model provider types.

    This enum represents different types of model providers or sources.

    Attributes:
        CLOUD_MODEL (str): Represents cloud-based model providers.
        HUGGING_FACE (str): Represents models from the Hugging Face platform.
        URL (str): Represents models accessible via a URL.
        DISK (str): Represents locally stored models on disk.
    """

    CLOUD_MODEL = "cloud_model"
    HUGGING_FACE = "hugging_face"
    URL = "url"
    DISK = "disk"


class UserRoleEnum(Enum):
    """Enumeration of user roles in the system.

    This enum defines the various roles that a user can have in the application.
    Each role represents a different level of access and permissions.

    Attributes:
        ADMIN (str): Administrator role with high-level permissions.
        SUPER_ADMIN (str): Super administrator role with the highest level of permissions.
        DEVELOPER (str): Role for software developers.
        DEVOPS (str): Role for DevOps engineers.
        TESTER (str): Role for quality assurance testers.
    """

    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"
    DEVELOPER = "developer"
    DEVOPS = "devops"
    TESTER = "tester"


class UserStatusEnum(StrEnum):
    """Enumeration of user statuses in the system.

    This enum defines the possible statuses that a user account can have.
    It uses auto() to automatically assign string values equal to the member names.

    Attributes:
        ACTIVE: Represents an active user account.
        INACTIVE: Represents an inactive or disabled user account.
        INVITED: Represents a user who has been invited but hasn't yet activated their account.
    """

    ACTIVE = auto()
    INACTIVE = auto()
    INVITED = auto()


class UserColorEnum(Enum):
    """Enumeration of predefined user colors.

    This enum defines a set of color options that can be assigned to users.
    Each color is represented by its hexadecimal code.

    Attributes:
        COLOR_1 (str): color (#E57333).
        COLOR_2 (str): color (#FFC442).
        COLOR_3 (str): color (#61A560).
        COLOR_4 (str): color (#3F8EF7).
        COLOR_5 (str): color (#C64C9C).
        COLOR_6 (str): color (#95E0FB).
    """

    COLOR_1 = "#E57333"
    COLOR_2 = "#FFC442"
    COLOR_3 = "#61A560"
    COLOR_4 = "#3F8EF7"
    COLOR_5 = "#C64C9C"
    COLOR_6 = "#95E0FB"

    @classmethod
    def get_random_color(cls) -> str:
        """Get a random color."""
        colors = list(cls)
        return random.choice(colors).value


class PermissionEnum(Enum):
    """Enumeration of system permissions.

    This enum defines various permission levels for different aspects of the system,
    including models, projects, endpoints, clusters, and user management.

    Attributes:
        MODEL_VIEW (str): Permission to view models.
        MODEL_MANAGE (str): Permission to manage models.
        MODEL_BENCHMARK (str): Permission to benchmark models.
        PROJECT_VIEW (str): Permission to view projects.
        PROJECT_MANAGE (str): Permission to manage projects.
        ENDPOINT_VIEW (str): Permission to view endpoints.
        ENDPOINT_MANAGE (str): Permission to manage endpoints.
        CLUSTER_VIEW (str): Permission to view clusters.
        CLUSTER_MANAGE (str): Permission to manage clusters.
        USER_MANAGE (str): Permission to manage users.
    """

    MODEL_VIEW = "model:view"
    MODEL_MANAGE = "model:manage"
    MODEL_BENCHMARK = "model:benchmark"

    PROJECT_VIEW = "project:view"
    PROJECT_MANAGE = "project:manage"

    ENDPOINT_VIEW = "endpoint:view"
    ENDPOINT_MANAGE = "endpoint:manage"

    CLUSTER_VIEW = "cluster:view"
    CLUSTER_MANAGE = "cluster:manage"

    USER_MANAGE = "user:manage"

    @classmethod
    def get_global_permissions(cls) -> List[str]:
        """Return all permission values in a list."""
        return [
            cls.MODEL_VIEW.value,
            cls.MODEL_MANAGE.value,
            cls.MODEL_BENCHMARK.value,
            cls.PROJECT_VIEW.value,
            cls.PROJECT_MANAGE.value,
            cls.CLUSTER_VIEW.value,
            cls.CLUSTER_MANAGE.value,
            cls.USER_MANAGE.value,
        ]

    @classmethod
    def get_default_permissions(cls) -> List[str]:
        """Return default permission values in a list."""
        return [
            cls.MODEL_VIEW.value,
            cls.MODEL_MANAGE.value,
            cls.PROJECT_VIEW.value,
            cls.CLUSTER_VIEW.value,
        ]

    @classmethod
    def get_protected_permissions(cls) -> List[str]:
        """Return restrictive permission values in a list."""
        return [
            cls.MODEL_VIEW.value,
            cls.PROJECT_VIEW.value,
            cls.CLUSTER_VIEW.value,
        ]

    @classmethod
    def get_project_default_permissions(cls) -> List[str]:
        """Return default permission values in a list."""
        return [
            cls.ENDPOINT_VIEW.value,
        ]

    @classmethod
    def get_project_level_scopes(cls) -> List[str]:
        """Return project-level scope values in a list."""
        return [
            cls.ENDPOINT_VIEW.value,
            cls.ENDPOINT_MANAGE.value,
        ]

    @classmethod
    def get_project_protected_scopes(cls) -> List[str]:
        """Return project-level protected scope values in a list."""
        return [
            cls.ENDPOINT_VIEW.value,
        ]


class TokenTypeEnum(Enum):
    """Enumeration of token types used in the application.

    This enum defines the different types of authentication tokens
    that can be used within the application.

    Attributes:
        ACCESS (str): Represents an access token.
        REFRESH (str): Represents a refresh token.
    """

    ACCESS = "access"
    REFRESH = "refresh"


# Algorithm used for signing tokens
JWT_ALGORITHM = "HS256"


class WorkflowStatusEnum(StrEnum):
    """Enumeration of workflow statuses."""

    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    # Cancelled status not required since workflow delete api delete record


class WorkflowTypeEnum(StrEnum):
    """Enumeration of workflow types."""

    MODEL_DEPLOYMENT = auto()
    MODEL_SECURITY_SCAN = auto()
    CLUSTER_ONBOARDING = auto()
    CLUSTER_DELETION = auto()
    ENDPOINT_DELETION = auto()
    CLOUD_MODEL_ONBOARDING = auto()
    LOCAL_MODEL_ONBOARDING = auto()


class NotificationType(Enum):
    """Represents the type of a notification.

    Attributes:
        EVENT: Notification triggered by an event.
        TOPIC: Notification related to a specific topic.
        BROADCAST: Notification triggered by a broadcast.
    """

    EVENT = "event"
    TOPIC = "topic"
    BROADCAST = "broadcast"


class NotificationCategory(str, Enum):
    """Represents the type of an internal notification.

    Attributes:
        INAPP: Represents the in-app notification type.
        INTERNAL: Represents the internal notification type.
    """

    INAPP = "inapp"
    INTERNAL = "internal"


class PayloadType(str, Enum):
    """Represents the type of a payload.

    Attributes:
        DEPLOYMENT_RECOMMENDATION: Represents the deployment recommendation payload type.
        DEPLOY_MODEL: Represents the model deployment payload type.
    """

    DEPLOYMENT_RECOMMENDATION = "get_cluster_recommendations"
    DEPLOY_MODEL = "deploy_model"
    REGISTER_CLUSTER = "register_cluster"
    DELETE_CLUSTER = "delete_cluster"
    DELETE_DEPLOYMENT = "delete_deployment"
    PERFORM_MODEL_EXTRACTION = "perform_model_extraction"
    PERFORM_MODEL_SECURITY_SCAN = "perform_model_security_scan"


class BudServeWorkflowStepEventName(str, Enum):
    """Represents the name of a workflow step event.

    Attributes:
        BUD_SIMULATOR_EVENTS: Represents the Bud simulator workflow step event name.
        BUDSERVE_CLUSTER_EVENTS: Represents the Budserve cluster workflow step event name.
        CREATE_CLUSTER_EVENTS: Represents the create cluster workflow step event name.
        MODEL_EXTRACTION_EVENTS: Represents the model extraction workflow step event name.
        MODEL_SECURITY_SCAN_EVENTS: Represents the model security scan workflow step event name.
    """

    BUD_SIMULATOR_EVENTS = "bud_simulator_events"
    BUDSERVE_CLUSTER_EVENTS = "budserve_cluster_events"
    CREATE_CLUSTER_EVENTS = "create_cluster_events"
    MODEL_EXTRACTION_EVENTS = "model_extraction_events"
    MODEL_SECURITY_SCAN_EVENTS = "model_security_scan_events"
    DELETE_CLUSTER_EVENTS = "delete_cluster_events"
    DELETE_ENDPOINT_EVENTS = "delete_endpoint_events"


class ClusterStatusEnum(StrEnum):
    """Cluster status types.

    Attributes:
        AVAILABLE: Represents the available cluster status.
        NOT_AVAILABLE: Represents the not available cluster status.
        REGISTERING: Represents the registering cluster status.
        ERROR: Represents the error cluster status.
    """

    AVAILABLE = auto()
    NOT_AVAILABLE = auto()
    REGISTERING = auto()
    ERROR = auto()
    DELETING = auto()
    DELETED = auto()


class EndpointStatusEnum(StrEnum):
    """Status for endpoint.

    Attributes:
        RUNNING: Represents the running endpoint status.
        FAILURE: Represents the failure endpoint status.
        DEPLOYING: Represents the deploying endpoint status.
        UNHEALTHY: Represents the unhealthy endpoint status.
        DELETING: Represents the deleting endpoint status.
    """

    RUNNING = auto()
    FAILURE = auto()
    DEPLOYING = auto()
    UNHEALTHY = auto()
    DELETING = auto()
    DELETED = auto()


class ModelTemplateTypeEnum(StrEnum):
    """Model template types."""

    SUMMARIZATION = auto()
    CHAT = auto()
    QUESTION_ANSWERING = auto()
    RAG = auto()
    CODE_GEN = auto()
    CODE_TRANSLATION = auto()
    ENTITY_EXTRACTION = auto()
    SENTIMENT_ANALYSIS = auto()
    DOCUMENT_ANALYSIS = auto()
    OTHER = auto()


class DropdownBackgroundColor(str, Enum):
    """Background hex color for dropdown."""

    COLOR_1 = "#EEEEEE"
    COLOR_2 = "#965CDE"
    COLOR_3 = "#EC7575"
    COLOR_4 = "#479D5F"
    COLOR_5 = "#D1B854"
    COLOR_6 = "#ECAE75"
    COLOR_7 = "#42CACF"
    COLOR_8 = "#DE5CD1"
    COLOR_9 = "#4077E6"
    COLOR_10 = "#8DE640"
    COLOR_11 = "#8E5EFF"
    COLOR_12 = "#FF895E"
    COLOR_13 = "#FF5E99"
    COLOR_14 = "#F4FF5E"
    COLOR_15 = "#FF5E5E"
    COLOR_16 = "#5EA3FF"
    COLOR_17 = "#5EFFBE"

    @classmethod
    def get_random_color(cls) -> str:
        """Get a random color."""
        colors = list(cls)
        return random.choice(colors).value


class BaseModelRelationEnum(StrEnum):
    """Base model relation types."""

    ADAPTER = "adapter"
    MERGE = "merge"
    QUANTIZED = "quantized"
    FINETUNE = "finetune"


class ModelSecurityScanStatusEnum(StrEnum):
    """Model security scan status types."""

    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()
    SAFE = auto()


LICENSE_DIR = "licenses"


class ModelStatusEnum(StrEnum):
    """Enumeration of entity statuses in the system.

    Attributes:
        ACTIVE: Represents an active entity.
        DELETED: Represents an deleted entity.
    """

    ACTIVE = auto()
    DELETED = auto()


class CloudModelStatusEnum(StrEnum):
    """Enumeration of entity statuses in the system.

    Attributes:
        ACTIVE: Represents an active entity.
        DELETED: Represents an deleted entity.
    """

    ACTIVE = auto()
    DELETED = auto()


class ProjectStatusEnum(StrEnum):
    """Enumeration of entity statuses in the system.

    Attributes:
        ACTIVE: Represents an active entity.
        DELETED: Represents an deleted entity.
    """

    ACTIVE = auto()
    DELETED = auto()


# Bud Notify Workflow
BUD_NOTIFICATION_WORKFLOW = "bud-notification"
BUD_INTERNAL_WORKFLOW = "bud-internal"


class NotificationStatus(Enum):
    """Enumerate notification statuses."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PENDING = "PENDING"
