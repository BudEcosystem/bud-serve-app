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

from typing import List, Union, Optional

from .schemas import ClusterListResponse, ClusterFilter
from .services import ClusterService

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