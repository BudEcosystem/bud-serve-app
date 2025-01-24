from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from budapp.commons import logging
from budapp.commons.dependencies import get_session
from budapp.commons.exceptions import ClientException

from ..commons.api_utils import pubsub_api_endpoint
from ..commons.schemas import ErrorResponse, SuccessResponse
from .crud import CredentialDataManager
from .models import Credential
from .schemas import CredentialUpdateRequest


logger = logging.get_logger(__name__)

credential_router = APIRouter(prefix="/credentials", tags=["credential"])


@credential_router.post("/update")
@pubsub_api_endpoint(request_model=CredentialUpdateRequest)
async def update_credential(
    credential_update_request: CredentialUpdateRequest,
    session: Annotated[Session, Depends(get_session)],
):
    """Update the credential last used at time."""
    logger.debug("Received request to subscribe to bud-serve-app credential update")
    try:
        payload = credential_update_request.payload
        logger.debug(f"Update CredentialReceived payload: {payload}")
        db_credential = await CredentialDataManager(session).retrieve_by_fields(Credential, {"hashed_key": payload.hashed_key})
        await CredentialDataManager(session).update_by_fields(db_credential, {"last_used_at": payload.last_used_at})
        return SuccessResponse(message="Credential updated successfully").to_http_response()
    except ClientException as e:
        logger.exception(f"Failed to execute credential update: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to update credential: {e}")
        return ErrorResponse(code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to update credential").to_http_response()
