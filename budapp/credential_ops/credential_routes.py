from typing import Annotated
from typing_extensions import Union

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from budapp.commons import logging
from budapp.commons.dependencies import get_session
from budapp.commons.exceptions import ClientException

from ..commons.api_utils import pubsub_api_endpoint
from ..commons.schemas import ErrorResponse, SuccessResponse
from .crud import CredentialDataManager, CredentialDataManager, CloudProviderDataManager
from .models import Credential, CloudProviders
from .schemas import CredentialUpdateRequest,CloudProvidersListResponse
from budapp.credential_ops.schemas import CloudProvidersCreateRequest


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
        db_credential = await CredentialDataManager(session).retrieve_by_fields(
            Credential, {"hashed_key": payload.hashed_key}
        )
        db_last_used_at = db_credential.last_used_at
        if db_last_used_at is None or db_last_used_at < payload.last_used_at:
            await CredentialDataManager(session).update_by_fields(
                db_credential, {"last_used_at": payload.last_used_at}
            )
        return SuccessResponse(message="Credential updated successfully").to_http_response()
    except ClientException as e:
        logger.exception(f"Failed to execute credential update: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to update credential: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to update credential"
        ).to_http_response()


# TODO: add cloud provider routes
@credential_router.get("/cloud-providers")
async def get_cloud_providers(
    session: Annotated[Session, Depends(get_session)],
):
    """Get all cloud providers."""
    logger.debug("Getting all the cloud providers")
    try:
        # Use CloudProviderDataManager to get all providers
        providers = await CloudProviderDataManager(session).get_all_providers()

        # Convert CloudCredentials objects to CloudProvidersSchema
        # This converts the list of CloudCredentials to list of CloudProvidersSchema
        provider_schemas = [provider.to_schema() for provider in providers]

        response = CloudProvidersListResponse(
            providers=provider_schemas,
            code=status.HTTP_200_OK,
            message="Cloud providers retrieved successfully"
        )
        return response.to_http_response()
    except Exception as e:
        logger.exception(f"Failed to get cloud providers: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Failed to get cloud providers"
        ).to_http_response()


@credential_router.post("/cloud-providers")
async def create_cloud_provider(
    cloud_provider_requst: CloudProvidersCreateRequest,
    session: Annotated[Session, Depends(get_session)],
):
    """Create a new cloud provider credential."""
    logger.debug(f"Creating cloud provider: {cloud_provider_requst}")

    # Validate the provider id in the database
    provider = await CloudProviderDataManager(session).retrieve_by_fields(
        CloudProviders,
        {"id": cloud_provider_requst.provider_id}
    )
    if not provider:
        raise ValueError("Provider ID is invalid")

    pass


@credential_router.put("/cloud-providers/{credential_id}")
async def update_cloud_provider(
    credential_id: str,
    cloud_provider: CloudProvider,
    session: Annotated[Session, Depends(get_session)],
):
    """Update a cloud provider credential_."""
    pass


@credential_router.delete("/cloud-providers/{credential_id}")
async def delete_cloud_provider(
    credential_id: str,
    session: Annotated[Session, Depends(get_session)],
):
    """Delete a cloud provider credential."""
    pass
