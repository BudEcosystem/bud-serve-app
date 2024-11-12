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

"""The cluster ops package, containing essential business logic, services, and routing configurations for the model ops."""

from typing import List, Union

from .schemas import ClusterResponse
from .services import ClusterService

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.commons import logging
from budapp.commons.dependencies import (
    get_current_active_user,
    get_session,
)
from budapp.commons.schemas import ErrorResponse
from budapp.user_ops.schemas import User

logger = logging.get_logger(__name__)

cluster_router = APIRouter(prefix="/clusters", tags=["cluster"])

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
            "model": ClusterResponse,
            "description": "Successfully listed all clusters",
        },
    },
    description="List all clusters",
)
async def list_clusters(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=0),
) -> Union[ClusterResponse, ErrorResponse]:
    """List all clusters."""
    # Calculate offset
    offset = (page - 1) * limit

    try:
        clusters, total_count = await ClusterService().get_all_clusters(offset, limit)
    except Exception as e:
        logger.exception(f"Failed to get all clusters: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get clusters"
        ).to_http_response()

    return ClusterResponse(
        clusters=clusters,
        total_record=total_count,
        page=page,
        limit=limit,
        object="cluster.list",
        code=status.HTTP_200_OK,
    ).to_http_response()
