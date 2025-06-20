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

"""The metric ops package, containing essential business logic, services, and routing configurations for the metric ops."""

from typing import Any, Dict, Union

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.commons import logging
from budapp.commons.dependencies import get_current_active_user, get_session
from budapp.commons.exceptions import ClientException
from budapp.commons.schemas import ErrorResponse
from budapp.user_ops.schemas import User

from .schemas import DashboardStatsResponse
from .services import BudMetricService, MetricService

logger = logging.get_logger(__name__)

metric_router = APIRouter(prefix="/metrics", tags=["metric"])


@metric_router.post(
    "/analytics",
    response_class=JSONResponse,
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
            "description": "Analytics response from metrics service",
        },
    },
    description="Proxy endpoint for analytics requests to the observability/analytics endpoint",
)
async def analytics_proxy(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    request_body: Dict[str, Any],
):
    """
    Proxy analytics requests to the observability/analytics endpoint.
    
    This endpoint forwards the request body to the metrics service
    and enriches the response with names for project, model, and endpoint IDs.
    """
    try:
        response_data = await BudMetricService(session).proxy_analytics_request(request_body)
        return JSONResponse(content=response_data, status_code=status.HTTP_200_OK)
    except ClientException as e:
        logger.exception(f"Failed to proxy analytics request: {e}")
        error_response = ErrorResponse(code=e.status_code, message=e.message)
        return JSONResponse(
            content=error_response.model_dump(mode="json"),
            status_code=e.status_code
        )
    except Exception as e:
        logger.exception(f"Failed to proxy analytics request: {e}")
        error_response = ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            message="Failed to proxy analytics request"
        )
        return JSONResponse(
            content=error_response.model_dump(mode="json"),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@metric_router.get(
    "/count",
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
            "model": DashboardStatsResponse,
            "description": "Successfully retrieved dashboard statistics",
        },
    },
    description="Retrieve the dashboard statistics, including counts for models, projects, endpoints, and clusters.",
)
async def get_dashboard_stats(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
) -> DashboardStatsResponse:
    """
    Retrieves the dashboard statistics, including counts for models, projects, endpoints, and clusters.

    Args:
        current_user (User): The current authenticated user making the request.
        session (Session): The database session used for querying data.

    Returns:
        DashboardStatsResponse: An object containing aggregated statistics for the dashboard,
        such as model counts, project counts, endpoint counts, and cluster counts.
    """
    try:
        return await MetricService(session).get_dashboard_stats(current_user.id)
    except ClientException as e:
        logger.exception(f"Failed to fetch dashboard statistics: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to fetch dashboard statistics: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to fetch dashboard statistics"
        ).to_http_response()