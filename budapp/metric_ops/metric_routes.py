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

from typing import Union

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from typing_extensions import Annotated

from budapp.commons import logging
from budapp.commons.dependencies import (
    get_current_active_user,
    get_session,
)
from budapp.commons.exceptions import ClientException
from budapp.commons.schemas import ErrorResponse
from budapp.user_ops.schemas import User

from .schemas import (
    CountAnalyticsRequest,
    CountAnalyticsResponse,
    PerformanceAnalyticsRequest,
    PerformanceAnalyticsResponse,
    DashboardStatsResponse,
)
from .services import MetricService


logger = logging.get_logger(__name__)

metric_router = APIRouter(prefix="/metrics", tags=["metric"])


@metric_router.post(
    "/analytics/request-counts",
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
            "model": CountAnalyticsResponse,
            "description": "Successfully get request count analytics",
        },
    },
    description="Get request count analytics",
)
async def get_request_count_analytics(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    metric_request: CountAnalyticsRequest,
) -> Union[CountAnalyticsResponse, ErrorResponse]:
    """Get request count analytics."""
    try:
        return await MetricService(session).get_request_count_analytics(metric_request)
    except ClientException as e:
        logger.exception(f"Failed to get request count analytics: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to get request count analytics: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get request count analytics"
        ).to_http_response()


@metric_router.post(
    "/analytics/request-performance",
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
            "model": PerformanceAnalyticsResponse,
            "description": "Successfully get request performance analytics",
        },
    },
    description="Get request performance analytics",
)
async def get_request_performance_analytics(
    current_user: Annotated[User, Depends(get_current_active_user)],
    session: Annotated[Session, Depends(get_session)],
    metric_request: PerformanceAnalyticsRequest,
) -> Union[PerformanceAnalyticsResponse, ErrorResponse]:
    """Get request performance analytics."""
    try:
        return await MetricService(session).get_request_performance_analytics(metric_request)
    except ClientException as e:
        logger.exception(f"Failed to get request performance analytics: {e}")
        return ErrorResponse(code=e.status_code, message=e.message).to_http_response()
    except Exception as e:
        logger.exception(f"Failed to get request performance analytics: {e}")
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get request performance analytics"
        ).to_http_response()


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
