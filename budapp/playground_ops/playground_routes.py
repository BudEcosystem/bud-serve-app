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

"""The playground ops package, containing essential business logic, services, and routing configurations for the playground ops."""

from typing import List, Optional, Union

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from ..commons import logging
from ..commons.dependencies import (
    get_session,
    parse_ordering_fields,
)
from ..commons.exceptions import ClientException
from ..commons.schemas import ErrorResponse
from .schemas import PlaygroundDeploymentFilter, PlaygroundDeploymentListResponse


logger = logging.get_logger(__name__)

playground_router = APIRouter(prefix="/playground", tags=["playground"])


@playground_router.get(
    "/deployments",
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
            "model": PlaygroundDeploymentListResponse,
            "description": "Successfully list all playground deployments",
        },
    },
    description="List all playground deployments",
)
async def list_playground_deployments(
    # current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    filters: Annotated[PlaygroundDeploymentFilter, Depends()],
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),
    search: bool = False,
) -> Union[PlaygroundDeploymentListResponse, ErrorResponse]:
    """List all playground deployments."""
    # Calculate offset
    offset = (page - 1) * limit

    # Convert PlaygroundDeploymentFilter to dictionary
    filters_dict = filters.model_dump(exclude_none=True)

    try:
        # db_endpoints, count = await EndpointService(session).get_all_endpoints(
        #     offset, limit, filters_dict, order_by, search
        # )
        db_endpoints = []
        count = 0
        return PlaygroundDeploymentListResponse(
            endpoints=db_endpoints,
            total_record=count,
            page=page,
            limit=limit,
            object="playground.deployments.list",
            code=status.HTTP_200_OK,
        ).to_http_response()
    except ClientException as e:
        logger.exception(f"Failed to get all playground deployments: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to get all playground deployments: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get all playground deployments"
        ).to_http_response()
