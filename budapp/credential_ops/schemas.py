from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator

from budapp.commons import logging
from budapp.commons.schemas import CloudEventBase


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
