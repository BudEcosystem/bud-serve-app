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
from .services import ClusterProviderService
from budapp.credential_ops.schemas import CloudProvidersSchema
import uuid

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


@credential_router.get("/cloud-providers")
async def get_cloud_providers(
    session: Annotated[Session, Depends(get_session)],
) -> Union[CloudProvidersListResponse, ErrorResponse]:
    """Get all cloud providers."""
    logger.debug("Getting all the cloud providers")
    try:
        # Use CloudProviderDataManager to get all providers
        providers = await CloudProviderDataManager(session).get_all_providers()

        # Convert SQLAlchemy objects to CloudProvidersSchema objects
        provider_schemas = []
        for provider in providers:
            # Extract attributes and handle type conversions
            provider_dict = {}
            for column in provider.__table__.columns:
                value = getattr(provider, column.name)

                # Convert specific types as needed
                if isinstance(value, uuid.UUID):
                    value = str(value)  # Convert UUID to string

                # Handle schema_definition specifically
                if column.name == 'schema_definition' and not isinstance(value, str):
                    import json
                    value = json.dumps(value)

                provider_dict[column.name] = value

            # Create schema object
            try:
                schema = CloudProvidersSchema(**provider_dict)
                provider_schemas.append(schema)
            except Exception as e:
                logger.error(f"Failed to create schema for provider {provider_dict.get('id', 'unknown')}: {e}")
                # Continue with next provider instead of failing the whole request
                continue

        # Create response
        response = CloudProvidersListResponse(
            providers=provider_schemas,
            code=status.HTTP_200_OK,
            message="Cloud providers retrieved successfully"
        )
        return response
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

    try:
        # Validate the provider id in the database
        provider = await CloudProviderDataManager(session).retrieve_by_fields(
            CloudProviders,
            {"id": cloud_provider_requst.provider_id}
        )
        if not provider:
            return ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
                message="Provider ID is invalid"
            ).to_http_response()

        # Save credentials via service
        await ClusterProviderService(session).create_provider_credential(cloud_provider_requst)

        return SuccessResponse(
            code=status.HTTP_201_CREATED,
            message="Cloud provider created successfully"
        ).to_http_response()

    except Exception as e:
        logger.exception(f"Failed to create cloud provider: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Failed to create cloud provider"
        ).to_http_response()


# @credential_router.put("/cloud-providers/{credential_id}")
# async def update_cloud_provider(
#     credential_id: str,
#     cloud_provider: CloudProviders,
#     session: Annotated[Session, Depends(get_session)],
# ):
#     """Update a cloud provider credential_."""
#     pass


# @credential_router.delete("/cloud-providers/{credential_id}")
# async def delete_cloud_provider(
#     credential_id: str,
#     session: Annotated[Session, Depends(get_session)],
# ):
#     """Delete a cloud provider credential."""
#     pass
