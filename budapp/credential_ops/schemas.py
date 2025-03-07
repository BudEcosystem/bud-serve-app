from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator

from budapp.commons import logging
from budapp.commons.schemas import CloudEventBase

from budapp.commons.schemas import SuccessResponse


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
    """Schema for cloud providers."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str
    description: str
    logo_url: str
    schema_definition: str


class CloudProvidersListResponse(SuccessResponse):
    """Response to list the cloud providers."""

    providers: list[CloudProvidersSchema]

class CloudProvidersCreateRequest(BaseModel):
    """Request to create a new cloud provider."""

    provider_id: str # TODO: Probably we need to use UUID, test and replace
    credential_values: dict[str, str] # JSON Structure


    # JSON Payload With Credential -  Depends on the provider & Validate
