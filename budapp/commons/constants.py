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

from enum import Enum, StrEnum, auto

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


class ModelTypeEnum(Enum):
    """Enumeration of model types.

    This enum represents different categories or types of AI models based on their primary function or training approach.

    Attributes:
        PRETRAINED (str): Represents models that have been pre-trained on large datasets and can be fine-tuned for specific tasks.
        CHAT (str): Represents models specifically designed or fine-tuned for conversational AI and chat applications.
        CODEGEN (str): Represents models specialized in code generation tasks.
    """

    PRETRAINED = "pretrained"
    CHAT = "chat"
    CODEGEN = "codegen"


ModelSourceEnum = create_dynamic_enum(
    "ModelSourceEnum",
    [
        "local",
        "nlp_cloud",
        "deepinfra",
        "anthropic",
        "vertex_ai-vision-models",
        "replicate",
        "vertex_ai-chat-models",
        "azure_ai",
        "perplexity",
        "vertex_ai-code-text-models",
        "vertex_ai-text-models",
        "palm",
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
