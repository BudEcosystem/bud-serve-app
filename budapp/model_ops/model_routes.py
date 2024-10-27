from typing import List, Optional, Union

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.commons import logging
from budapp.commons.dependencies import (
    get_current_active_user,
    get_session,
    parse_ordering_fields,
)
from budapp.commons.schemas import ErrorResponse
from budapp.user_ops.schemas import User

from .schemas import ProviderFilter, ProviderResponse
from .services import ProviderService

logger = logging.get_logger(__name__)

model_router = APIRouter(prefix="/models", tags=["model"])


@model_router.get(
    "/providers",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to server error",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": ErrorResponse,
            "description": "Service is unavailable due to client error",
        },
        status.HTTP_200_OK: {
            "model": ProviderResponse,
            "description": "Successfully list all providers",
        },
    },
    description="List all model providers",
)
async def list_providers(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    filters: ProviderFilter = Depends(),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),
    search: bool = False,
) -> Union[ProviderResponse, ErrorResponse]:
    """List all model providers."""
    # Calculate offset
    offset = (page - 1) * limit

    # Convert UserFilter to dictionary
    filters_dict = filters.model_dump(exclude_none=True)

    try:
        db_providers, count = await ProviderService(session).get_all_providers(
            offset, limit, filters_dict, order_by, search
        )
    except Exception as e:
        logger.exception(f"Failed to get all providers: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get all providers"
        ).to_http_response()

    return ProviderResponse(
        providers=db_providers,
        total_record=count,
        page=page,
        limit=limit,
        object="providers.list",
        code=status.HTTP_200_OK,
    ).to_http_response()
