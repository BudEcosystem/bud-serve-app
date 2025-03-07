from typing import Annotated
from typing_extensions import Union

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from budapp.commons import logging
from budapp.commons.dependencies import (
    get_session,
    get_current_active_user
)
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
import json
from budapp.user_ops.schemas import User
from budapp.credential_ops.schemas import CloudCredentialResponse
from budapp.credential_ops.crud import CloudProviderCredentialDataManager
from budapp.credential_ops.schemas import CloudCredentialSchema

from fastapi import Path
from uuid import UUID
from budapp.credential_ops.models import CloudCredentials

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
            provider_dict = {
                column.name: str(getattr(provider, column.name)) if isinstance(getattr(provider, column.name), uuid.UUID)
                else getattr(provider, column.name)
                for column in provider.__table__.columns
            }

            # Ensure schema_definition is a valid JSON string
            if "schema_definition" in provider_dict and not isinstance(provider_dict["schema_definition"], str):
                provider_dict["schema_definition"] = json.dumps(provider_dict["schema_definition"])

            # Ensure `id` is always a string before validation
            if isinstance(provider_dict.get("id"), uuid.UUID):
                provider_dict["id"] = str(provider_dict["id"])

            try:
                schema = CloudProvidersSchema(**provider_dict)
                provider_schemas.append(schema)
            except Exception as e:
                logger.error(f"Failed to create schema for provider {provider_dict.get('id', 'unknown')}: {e}")
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
    current_user: Annotated[User, Depends(get_current_active_user)],
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
        await ClusterProviderService(session).create_provider_credential(cloud_provider_requst, current_user.id)

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

@credential_router.get("/cloud-providers/credentials", response_model=CloudCredentialResponse)
async def get_user_cloud_credentials(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
):
    """
    Retrieve all cloud provider credentials for the current user.

    Returns:
        CloudCredentialResponse: List of cloud provider credentials for the user.
    """
    logger.debug(f"Retrieving cloud credentials for user: {current_user.id}")

    try:
        # Use the CloudProviderCredentialDataManager to get user credentials
        credentials = await CloudProviderCredentialDataManager(session).get_credentials_by_user(current_user.id)

        # Convert database models to response schemas
        credential_schemas = []
        for cred in credentials:
            # Get provider information
            provider = await CloudProviderDataManager(session).retrieve_by_fields(
                CloudProviders, {"id": cred.provider_id}
            )

            # Create credential schema with masked sensitive data
            credential_schema = CloudCredentialSchema(
                id=str(cred.id),
                provider_id=str(cred.provider_id),
                provider_name=provider.name if provider else "Unknown Provider",
                created_at=cred.created_at,
                # Mask sensitive information in credential values
                credential_summary=_get_masked_credential_summary(cred.credential, provider.schema_definition if provider else {})
            )
            credential_schemas.append(credential_schema)

        return CloudCredentialResponse(
            credentials=credential_schemas,
            code=status.HTTP_200_OK,
            message="Cloud provider credentials retrieved successfully"
        )
    except Exception as e:
        logger.exception(f"Failed to retrieve cloud credentials: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Failed to retrieve cloud provider credentials"
        ).to_http_response()

@credential_router.get("/cloud-providers/credentials/{credential_id}", response_model=CloudCredentialResponse)
async def get_user_cloud_credential(
    credential_id: Annotated[str, Path(title="The ID of the credential to retrieve")],
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
):
    """
    Retrieve a specific cloud provider credential for the current user.

    Args:
        credential_id: ID of the credential to retrieve

    Returns:
        CloudCredentialResponse: The requested cloud provider credential.
    """
    logger.debug(f"Retrieving cloud credential {credential_id} for user: {current_user.id}")

    try:
        # Convert string ID to UUID
        try:
            cred_uuid = UUID(credential_id)
        except ValueError:
            return ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
                message="Invalid credential ID format"
            ).to_http_response()

        # Query the specific credential
        credential = await CloudProviderDataManager(session).retrieve_by_fields(
            CloudCredentials, {"id": cred_uuid}
        )

        if not credential:
            return ErrorResponse(
                code=status.HTTP_404_NOT_FOUND,
                message="Credential not found"
            ).to_http_response()

        # Verify the credential belongs to the current user
        if credential.user_id != current_user.id:
            return ErrorResponse(
                code=status.HTTP_403_FORBIDDEN,
                message="You don't have permission to access this credential"
            ).to_http_response()

        # Get provider information
        provider = await CloudProviderDataManager(session).retrieve_by_fields(
            CloudProviders, {"id": credential.provider_id}
        )

        # Create credential schema
        credential_schema = CloudCredentialSchema(
            id=str(credential.id),
            provider_id=str(credential.provider_id),
            provider_name=provider.name if provider else "Unknown Provider",
            created_at=credential.created_at,
            # For individual credential view, provide more detailed but still masked info
            credential_summary=_get_masked_credential_summary(
                credential.credential,
                provider.schema_definition if provider else {},
                detailed=True
            )
        )

        return CloudCredentialResponse(
            credentials=[credential_schema],
            code=status.HTTP_200_OK,
            message="Cloud provider credential retrieved successfully"
        )
    except Exception as e:
        logger.exception(f"Failed to retrieve cloud credential: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message="Failed to retrieve cloud provider credential"
        ).to_http_response()


# Helper function for masking sensitive data
def _get_masked_credential_summary(credential_values: dict, schema_definition: dict, detailed: bool = False) -> dict:
    """
    Create a summary of credential values with sensitive information masked.

    Args:
        credential_values: The raw credential values
        schema_definition: The schema definition from the provider
        detailed: Whether to include more details (for individual credential view)

    Returns:
        Dict with masked sensitive values
    """
    result = {}

    # If we don't have a schema definition, just mask everything
    if not schema_definition:
        return {k: "********" for k in credential_values.keys()}

    # Get the schema properties
    properties = schema_definition.get("properties", {})

    for key, value in credential_values.items():
        property_info = properties.get(key, {})
        # Check if this is a secret field (look for hints in the schema)
        is_secret = (
            property_info.get("format") == "password" or
            property_info.get("x-sensitive") == True or
            "secret" in key.lower() or
            "password" in key.lower() or
            "key" in key.lower() or
            "token" in key.lower()
        )

        if is_secret:
            # For secrets, show only "********"
            result[key] = "********"
        elif detailed:
            # For detailed view, include non-sensitive values
            result[key] = value
        else:
            # For list view, include a simplified view
            if isinstance(value, str) and len(value) > 10:
                result[key] = value[:4] + "..." + value[-4:]
            else:
                result[key] = value

    return result
