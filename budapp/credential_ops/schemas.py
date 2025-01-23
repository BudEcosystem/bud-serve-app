from datetime import datetime
from pydantic import BaseModel
from budapp.commons.schemas import CloudEventBase


class CredentialUpdatePayload(BaseModel):
    hashed_key: str
    last_used_at: datetime

class CredentialUpdateRequest(CloudEventBase):
    payload: CredentialUpdatePayload
    
