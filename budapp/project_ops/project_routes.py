#  -----------------------------------------------------------------------------
#  Copyright (c) 2024 Bud Ecosystem Inc.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  -----------------------------------------------------------------------------

"""The project ops package, containing essential business logic, services, and routing configurations for the project ops."""

from typing import List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.commons import logging
from budapp.commons.dependencies import (
    get_current_active_user,
    get_session,
    parse_ordering_fields,
)
from budapp.commons.exceptions import ClientException
from budapp.commons.schemas import ErrorResponse
from budapp.user_ops.schemas import User

from .schemas import EditProjectRequest, ProjectClusterFilter, ProjectClusterPaginatedResponse, SingleProjectResponse
from .services import ProjectService


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


@project_router.get(
    "/{project_id}/clusters",
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
            "model": ProjectClusterPaginatedResponse,
            "description": "Successfully list all clusters in a project",
        },
    },
    description="List all clusters in a project.\n\nOrder by values are: name, endpoint_count, hardware_type, status, created_at, modified_at",
)
async def list_all_clusters(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    filters: Annotated[ProjectClusterFilter, Depends()],
    project_id: UUID,
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),
    search: bool = False,
) -> Union[ProjectClusterPaginatedResponse, ErrorResponse]:
    """List all clusters in a project."""
    # Calculate offset
    offset = (page - 1) * limit

    # Construct filters
    filters_dict = filters.model_dump(exclude_none=True, exclude_unset=True)

    try:
        result, count = await ProjectService(session).get_all_clusters_in_project(
            project_id, offset, limit, filters_dict, order_by, search
        )
    except ClientException as e:
        logger.exception(f"Failed to get all clusters: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to get all clusters: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get all clusters"
        ).to_http_response()

    return ProjectClusterPaginatedResponse(
        clusters=result,
        total_record=count,
        page=page,
        limit=limit,
        object="project.clusters.list",
        code=status.HTTP_200_OK,
        message="Successfully list all clusters in a project",
    ).to_http_response()
