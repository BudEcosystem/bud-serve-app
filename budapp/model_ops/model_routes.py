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

from .schemas import (
    AddCloudModelWorkflowRequest,
    AddCloudModelWorkflowResponse,
    ProviderFilter,
    ProviderResponse,
)
from .services import ModelService, ProviderService


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


@model_router.post(
    "/cloud-model-workflow",
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
            "model": AddCloudModelWorkflowResponse,
            "description": "Successfully add cloud model workflow",
        },
    },
    description="Add cloud model workflow",
)
async def add_cloud_model_workflow(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    request: AddCloudModelWorkflowRequest,
) -> Union[AddCloudModelWorkflowResponse, ErrorResponse]:
    """Add cloud model workflow."""
    try:
        db_workflow = await ModelService(session).add_cloud_model_workflow(
            current_user_id=current_user.id,
            step_number=request.step_number,
            workflow_id=request.workflow_id,
            workflow_total_steps=request.workflow_total_steps,
            provider_type=request.provider_type,
            source=request.source,
            name=request.name,
            modality=request.modality,
            uri=request.uri,
            tags=request.tags,
            icon=request.icon,
            trigger_workflow=request.trigger_workflow,
        )
    except Exception as e:
        logger.exception(f"Failed to get all providers: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get all providers"
        ).to_http_response()

    return AddCloudModelWorkflowResponse(
        workflow_id=db_workflow.id,
        status=db_workflow.status,
        current_step=db_workflow.current_step,
        total_steps=db_workflow.total_steps,
        reason=db_workflow.reason,
        workflow_steps=db_workflow.steps,
    ).to_http_response()
