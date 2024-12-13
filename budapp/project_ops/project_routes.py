from budapp.commons import logging
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.commons.schemas import ErrorResponse
from budapp.commons.exceptions import ClientException
from budapp.commons.dependencies import (
    get_current_active_user,
    get_session,
    parse_ordering_fields,
)
from budapp.user_ops.schemas import User
from .schemas import EditProjectRequest, SingleProjectResponse
from .services import ProjectService
from typing import List, Union
from uuid import UUID

logger = logging.get_logger(__name__)

project_router = APIRouter(prefix="/projects", tags=["project"])


@project_router.patch(
    "/{project_id}",
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
            "model": SingleProjectResponse,
            "description": "Successfully edited project",
        },
    },
    description="Edit project",
)
async def edit_project(
    project_id: UUID,
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    edit_project: EditProjectRequest,
) -> Union[SingleProjectResponse, ErrorResponse]:
    """Edit project"""
    try:
        db_project = await ProjectService(session).edit_project(
            project_id=project_id, data=edit_project.model_dump(exclude_unset=True, exclude_none=True)
        )
        return SingleProjectResponse(
            project=db_project,
            message="Project details updated successfully",
            code=status.HTTP_200_OK,
            object="project.edit",
        )
    except ClientException as e:
        logger.exception(f"Failed to edit project: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to edit project: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to edit project"
        ).to_http_response()
