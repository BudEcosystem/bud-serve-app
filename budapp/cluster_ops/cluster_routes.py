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

"""The cluster ops package, containing essential business logic, services, and routing configurations for the cluster ops."""

from typing import List, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile, status
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
from budapp.workflow_ops.schemas import RetrieveWorkflowDataResponse
from budapp.workflow_ops.services import WorkflowService

from .schemas import ClusterFilter, ClusterListResponse, CreateClusterWorkflowRequest
from .services import ClusterService


logger = logging.get_logger(__name__)

cluster_router = APIRouter(prefix="/clusters", tags=["cluster"])


@cluster_router.post(
    "/clusters",
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
            "model": RetrieveWorkflowDataResponse,
            "description": "Create cluster workflow executed successfully",
        },
    },
    description="Create cluster workflow",
)
async def create_cluster_workflow(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    step_number: Annotated[int, Form(gt=0)],
    name: Annotated[str | None, Form(min_length=1, max_length=100)] = None,
    ingress_url: Annotated[str | None, Form(min_length=1, max_length=100)] = None,
    configuration_file: Annotated[
        UploadFile | None, File(description="The configuration file for the cluster")
    ] = None,
    workflow_id: Annotated[UUID | None, Form()] = None,
    workflow_total_steps: Annotated[int | None, Form()] = None,
    trigger_workflow: Annotated[bool, Form()] = False,
) -> Union[RetrieveWorkflowDataResponse, ErrorResponse]:
    """Create cluster workflow."""
    # Perform router level validation
    if workflow_id is None and workflow_total_steps is None:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            message="workflow_total_steps is required when workflow_id is not provided",
        ).to_http_response()

    if workflow_id is not None and workflow_total_steps is not None:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            message="workflow_total_steps and workflow_id cannot be provided together",
        ).to_http_response()

    # Check if at least one of the other fields is provided
    other_fields = [name, ingress_url, configuration_file]
    required_fields = ["name", "ingress_url", "configuration_file"]
    if not any(other_fields):
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            message=f"At least one of {', '.join(required_fields)} is required when workflow_id is provided",
        )

    try:
        db_workflow = await ClusterService(session).create_cluster_workflow(
            current_user_id=current_user.id,
            request=CreateClusterWorkflowRequest(
                name=name,
                ingress_url=ingress_url,
                workflow_id=workflow_id,
                workflow_total_steps=workflow_total_steps,
                step_number=step_number,
                trigger_workflow=trigger_workflow,
            ),
            configuration_file=configuration_file,
        )

        return await WorkflowService(session).retrieve_workflow_data(db_workflow.id)
    except ClientException as e:
        logger.exception(f"Failed to execute create cluster workflow: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.error(f"Error occurred while executing create cluster workflow: {str(e)}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to execute create cluster workflow"
        ).to_http_response()


@cluster_router.get(
    "/clusters",
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
            "model": ClusterListResponse,
            "description": "Successfully listed all clusters",
        },
    },
    description="List all clusters",
)
async def list_clusters(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    filters: ClusterFilter = Depends(),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),
    search: bool = False,
) -> Union[ClusterListResponse, ErrorResponse]:
    """List all clusters."""
    offset = (page - 1) * limit

    filters_dict = filters.model_dump(exclude_none=True)

    try:
        db_clusters, count = await ClusterService(session).get_all_active_clusters(
            offset, limit, filters_dict, order_by, search
        )
    except Exception as e:
        logger.error(f"Error occurred while listing clusters: {str(e)}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to list clusters"
        ).to_http_response()

    return ClusterListResponse(
        clusters=db_clusters,
        total_record=count,
        page=page,
        limit=limit,
        object="cluster.list",
        code=status.HTTP_200_OK,
    ).to_http_response()
