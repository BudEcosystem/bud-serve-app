from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator, Field, field_validator

from budapp.commons import logging
from budapp.commons.schemas import CloudEventBase

from budapp.commons.schemas import SuccessResponse

import json
from typing import Dict, Any, Union
import uuid

logger = logging.get_logger(__name__)


class CredentialUpdatePayload(BaseModel):
    hashed_key: str
    last_used_at: datetime

class CredentialUpdateRequest(CloudEventBase):
    """Request to update the credential last used at time."""

    model_config = ConfigDict(extra="allow")

    payload: CredentialUpdatePayload

    @model_validator(mode="before")
    def log_credential_update(cls, data):
        """Log the credential update hits for debugging purposes."""
        # TODO: remove this function after Debugging
        logger.info("================================================")
        logger.info("Received hit in credentials/update:")
        logger.info(f"{data}")
        logger.info("================================================")
        return data


# Cloud Providers
class CloudProvidersSchema(BaseModel):
    """
    Schema for cloud providers.

    Attributes:
        id: Unique identifier for the cloud provider
        name: Display name of the cloud provider
        description: Detailed description of the cloud provider
        logo_url: URL to the provider's logo image
        schema_definition: JSON schema defining the configuration options for this provider
    """

    model_config = ConfigDict(extra="allow")

    id: Union[str, uuid.UUID] = Field(..., description="Unique identifier for the cloud provider")
    name: str = Field(..., description="Display name of the cloud provider")
    description: str = Field(..., description="Detailed description of the cloud provider")
    logo_url: str = Field(..., description="URL to the provider's logo image")
    schema_definition: Union[str, Dict[str, Any]] = Field(
        ...,
        description="JSON schema defining the configuration options for this provider"
    )

    @field_validator('schema_definition')
    @classmethod
    def validate_schema_definition(cls, v: Union[str, Dict[str, Any]]) -> str:
        """
        Validates the schema_definition field, ensuring it's in the correct format.

        If a dictionary is provided, it's converted to a JSON string.
        If a string is provided, it's validated as valid JSON.

        Args:
            v: The schema definition value to validate

        Returns:
            A JSON string representing the schema definition

        Raises:
            ValueError: If the provided value is not valid JSON
        """
        if isinstance(v, dict):
            return json.dumps(v)
        elif isinstance(v, str):
            try:
                # Validate string is proper JSON by trying to parse it
                json.loads(v)
                return v
            except json.JSONDecodeError:
                raise ValueError("schema_definition must be valid JSON")
        else:
            raise ValueError("schema_definition must be a JSON string or a dictionary")

    @model_validator(mode='after')
    def validate_schema(self) -> 'CloudProvidersSchema':
        """
        Validates the entire model after all fields have been processed.

        This can be used for cross-field validation or additional schema validation.

        Returns:
            The validated model
        """
        # Example: Ensure the schema is valid by checking if it can be parsed as JSON
        try:
            if isinstance(self.schema_definition, str):
                schema_dict = json.loads(self.schema_definition)
                # Add any additional schema validation logic here
                # For example, check if required fields exist
                if 'type' not in schema_dict:
                    raise ValueError("Schema must specify a 'type' field")
        except Exception as e:
            raise ValueError(f"Invalid schema_definition: {str(e)}")
        return self

    def get_schema_as_dict(self) -> Dict[str, Any]:
        """
        Helper method to get the schema definition as a Python dictionary.

        Returns:
            The schema definition as a dictionary
        """
        if isinstance(self.schema_definition, str):
            return json.loads(self.schema_definition)
        return self.schema_definition


class CloudProvidersListResponse(SuccessResponse):
    """Response to list the cloud providers."""

    providers: list[CloudProvidersSchema]

class CloudProvidersCreateRequest(BaseModel):
    """Request to create a new cloud provider."""

    provider_id: str # TODO: Probably we need to use UUID, test and replace
    credential_values: dict[str, str] # JSON Structure


class CloudCredentialSchema(BaseModel):
    """
    Schema for cloud credential.

    Attributes:
        id: Unique identifier for the credential
        provider_id: ID of the provider this credential is for
        provider_name: Display name of the provider
        created_at: When the credential was created
        credential_summary: Summary of credential values with sensitive information masked
    """
    id: str = Field(..., description="Unique identifier for the credential")
    provider_id: str = Field(..., description="ID of the provider this credential is for")
    provider_name: str = Field(..., description="Display name of the provider")
    created_at: datetime = Field(..., description="When the credential was created")
    credential_summary: Dict[str, Any] = Field(
        ...,
        description="Summary of credential values with sensitive information masked"
    )

class CloudCredentialResponse(SuccessResponse):
    """Response containing cloud credentials."""
    credentials: list[CloudCredentialSchema]
