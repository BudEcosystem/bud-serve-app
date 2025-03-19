from typing import Annotated, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from ..commons import logging
from ..commons.api_utils import pubsub_api_endpoint
from ..commons.constants import CredentialTypeEnum
from ..commons.dependencies import get_current_active_user, get_session, parse_ordering_fields
from ..commons.exceptions import ClientException
from ..commons.schemas import (
    ErrorResponse,
    SingleResponse,
    SuccessResponse,
)
from ..commons.security import RSAHandler
from ..user_ops.schemas import User
from .crud import CredentialDataManager
from .models import Credential
from .schemas import (
    PROPRIETARY_CREDENTIAL_DATA,
    CredentialDetails,
    CredentialFilter,
    CredentialRequest,
    CredentialResponse,
    CredentialUpdate,
    CredentialUpdateRequest,
    PaginatedCredentialResponse,
    ProprietaryCredentialDetailedView,
    ProprietaryCredentialFilter,
    ProprietaryCredentialRequest,
    ProprietaryCredentialResponse,
    ProprietaryCredentialUpdate,
    RouterConfig,
)
from .services import CredentialService, ProprietaryCredentialService


logger = logging.get_logger(__name__)

credential_router = APIRouter(prefix="/credentials", tags=["credential"])
proprietary_credential_router = APIRouter(prefix="/proprietary/credentials", tags=["proprietary credential"])
error_responses = {
    401: {"model": ErrorResponse},
    422: {"model": ErrorResponse},
}


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


@credential_router.get(
    "/router-config",
    response_model=SingleResponse[RouterConfig],
    responses=error_responses,
    description="Get router config for the given API key and endpoint name",
)
async def get_router_config(api_key: str, endpoint_name: str, session: Annotated[Session, Depends(get_session)]):
    router_config = await CredentialService(session).get_router_config(api_key, endpoint_name)
    return SingleResponse(message="Router config retrieved successfully", result=router_config)


@credential_router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=SingleResponse[CredentialResponse],
    responses={
        **error_responses,
        500: {"model": ErrorResponse},
    },
    description=f"""Add or generate a new credential for user. Valid credential types: 
    {", ".join([value.value for value in CredentialTypeEnum])}.
    For budserve credential type, project_id, expiry(None or 30, 60) are required.""",
)
async def add_credential(
    credential: CredentialRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
):
    credential_response = await CredentialService(session).add_credential(current_user.id, credential)
    logger.info(f"API-Key credential added: {credential_response.key}")

    return SingleResponse(message="Credential added successfully", result=credential_response)


@credential_router.get(
    "/",
    response_model=PaginatedCredentialResponse,
    responses=error_responses,
    description="Get saved credentials of user",
)
async def retrieve_credentials(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],  # noqa: B008
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    filters: CredentialFilter = Depends(),  # noqa: B008
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),  # noqa: B008
    search: bool = False,
):
    # Calculate offset
    offset = (page - 1) * limit

    # Convert Filter to dictionary
    filters_dict = filters.model_dump(exclude_none=True)
    filters_dict["user_id"] = current_user.id
    results, count = await CredentialService(session).get_credentials(offset, limit, filters_dict, order_by, search)

    return PaginatedCredentialResponse(
        message="Credentials listed successfully",
        credentials=results,
        total_record=count,
        page=page,
        limit=limit,
    )


@credential_router.put(
    "/{credential_id}",
    response_model=SingleResponse[CredentialResponse],
    response_model_exclude_none=True,
    responses={
        **error_responses,
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    description="Update saved credential of user.",
)
async def update_credential(
    credential_id: UUID,
    credential_data: CredentialUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
):
    db_credential = await CredentialService(session).update_credential(credential_data, credential_id, current_user.id)
    logger.info(f"Credential updated: {db_credential.id}")

    credential_response = CredentialResponse(
        name=db_credential.name,
        project_id=db_credential.project_id,
        key=await RSAHandler().encrypt(db_credential.key),
        expiry=db_credential.expiry,
        max_budget=db_credential.max_budget,
        model_budgets=db_credential.model_budgets,
        id=db_credential.id,
    )

    return SingleResponse(message="Credential updated successfully", result=credential_response)


@credential_router.delete(
    "/{credential_id}",
    response_model=SuccessResponse,
    responses={
        **error_responses,
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    description="Delete saved credential of user",
)
async def delete_credential(
    credential_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
):
    await CredentialService(session).delete_credential(credential_id, current_user.id)
    logger.info("Credential deleted")

    return SuccessResponse(message="Credential deleted successfully")


@credential_router.get(
    "/details/{api_key}",
    response_model=SingleResponse[CredentialDetails],
    responses=error_responses,
    description="Get credential details for the given API key",
)
async def retrieve_credential_details(
    api_key: str,
    session: Annotated[Session, Depends(get_session)],
):
    credential_detail = await CredentialService(session).retrieve_credential_details(api_key)
    logger.info("Credentials fetched successfully")

    return SingleResponse(message="Credentials retrieved successfully", result=credential_detail)


@credential_router.get(
    "/decrypt/{api_key}",
    response_model=SingleResponse[str],
    responses=error_responses,
    description="Get credential details for the given API key",
)
async def decrypt_credential(
    api_key: str,
    session: Annotated[Session, Depends(get_session)],
):
    decrypted_key = await CredentialService(session).decrypt_credential(api_key)
    logger.info("Credentials decrypted successfully")

    return SingleResponse(message="Credentials decrypted successfully", result=decrypted_key)


@proprietary_credential_router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    response_model=SingleResponse[ProprietaryCredentialResponse],
    responses={
        **error_responses,
        500: {"model": ErrorResponse},
    },
    description=f"""Add or generate a new credential for proprietary models.
    Valid credential types: {", ".join([value.value for value in CredentialTypeEnum])}.
    For budserve credential type, project_id, expiry(None or 30, 60) are required.""",
)
async def add_proprietary_credential(
    credential: ProprietaryCredentialRequest,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
):
    credential_response = await ProprietaryCredentialService(session).add_credential(current_user.id, credential)
    logger.info(f"{credential.type.value} credential added: {credential_response}")

    return SingleResponse(message="Credential added successfully", result=credential_response)


@proprietary_credential_router.get(
    "/",
    response_model=PaginatedCredentialResponse,
    responses=error_responses,
    description="Get proprietary credentials of user",
)
async def retrieve_proprietary_credentials(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],  # noqa: B008
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    filters: ProprietaryCredentialFilter = Depends(),  # noqa: B008
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),  # noqa: B008
    search: bool = False,
):
    # Calculate offset
    offset = (page - 1) * limit

    # Convert Filter to dictionary
    filters_dict = filters.model_dump(exclude_none=True)
    filters_dict["user_id"] = current_user.id
    results, count = await ProprietaryCredentialService(session).get_all_credentials(
        offset, limit, filters_dict, order_by, search
    )

    return PaginatedCredentialResponse(
        message="Proprietary credentials listed successfully",
        credentials=results,
        total_record=count,
        page=page,
        limit=limit,
    )


@proprietary_credential_router.put(
    "/{credential_id}",
    response_model=SingleResponse[ProprietaryCredentialResponse],
    response_model_exclude_none=True,
    responses={
        **error_responses,
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    description="Update saved proprietary credential of user.",
)
async def update_proprietary_credential(
    credential_id: UUID,
    credential_data: ProprietaryCredentialUpdate,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
):
    db_credential = await ProprietaryCredentialService(session).update_credential(
        credential_id, credential_data, current_user.id
    )
    logger.info(f"Credential updated: {db_credential.id}")

    return SingleResponse(message="Credential updated successfully", result=db_credential)


@proprietary_credential_router.delete(
    "/{credential_id}",
    response_model=SuccessResponse,
    responses={
        **error_responses,
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    description="Delete saved proprietary credential of user",
)
async def delete_proprietary_credential(
    credential_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
):
    await ProprietaryCredentialService(session).delete_credential(credential_id, current_user.id)
    logger.info("Credential deleted")

    return SuccessResponse(message="Credential deleted successfully")


@proprietary_credential_router.get(
    "/provider-info",
    response_model=SingleResponse,
    responses={
        **error_responses,
        500: {"model": ErrorResponse},
    },
    description="Different proprietary provider information",
)
async def get_provider_info(
    current_user: Annotated[User, Depends(get_current_active_user)],
    provider_name: CredentialTypeEnum,
):
    result = PROPRIETARY_CREDENTIAL_DATA.get(
        provider_name.value if provider_name else None, PROPRIETARY_CREDENTIAL_DATA
    )
    return SingleResponse(
        message="Provider info retrieved successfully",
        result=result,
    )


@proprietary_credential_router.get(
    "/{credential_id}/detailed-view",
    response_model=SingleResponse[ProprietaryCredentialDetailedView],
    responses=error_responses,
    description="Get details of a proprietary credential",
)
async def get_proprietary_credential_detailed_view(
    credential_id: UUID,
    _: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
):
    credential_details = await ProprietaryCredentialService(session).get_credential_details(
        credential_id, detailed_view=True
    )
    return SingleResponse(message="Credential details fetched successfully", result=credential_details)


@proprietary_credential_router.get(
    "/{credential_id}/details",
    response_model=SingleResponse[ProprietaryCredentialResponse],
    responses=error_responses,
    description="Get details of a proprietary credential",
)
async def get_proprietary_credential_details(credential_id: UUID, session: Annotated[Session, Depends(get_session)]):
    credential_details = await ProprietaryCredentialService(session).get_credential_details(credential_id)
    return SingleResponse(message="Credential details fetched successfully", result=credential_details)
