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

from typing import List, Literal, Optional, Union
from uuid import UUID

from fastapi import APIRouter, Depends, status
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

from .schemas import (
    CacheMetricsResponse,
    CountAnalyticsRequest,
    CountAnalyticsResponse,
    DashboardStatsResponse,
    InferenceQualityAnalyticsPromptFilter,
    InferenceQualityAnalyticsPromptResponse,
    InferenceQualityAnalyticsResponse,
    PerformanceAnalyticsRequest,
    PerformanceAnalyticsResponse,
)
from .services import BudMetricService, MetricService


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
    # session: Annotated[Session, Depends(get_session)],
    metric_request: CountAnalyticsRequest,
) -> Union[CountAnalyticsResponse, ErrorResponse]:
    """Get request count analytics."""
    try:
        return await BudMetricService().get_request_count_analytics(metric_request)
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
    # session: Annotated[Session, Depends(get_session)],
    metric_request: PerformanceAnalyticsRequest,
) -> Union[PerformanceAnalyticsResponse, ErrorResponse]:
    """Get request performance analytics."""
    try:
        return await BudMetricService().get_request_performance_analytics(metric_request)
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

@metric_router.post(
    "/analytics/cache-metrics/{endpoint_id}",
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
            "model": CacheMetricsResponse,
            "description": "Successfully get request performance analytics",
        },
    },
    description="Get deployment cache metrics",
)
async def get_deployment_cache_metric(
    endpoint_id: UUID,
    _: Annotated[User, Depends(get_current_active_user)],
    # session: Annotated[Session, Depends(get_session)],
    page: int = 1,
    limit: int = 10,
) -> Union[CacheMetricsResponse, ErrorResponse]:
    """Get deployment cache metrics."""
    try:
        response = await BudMetricService().get_deployment_cache_metric(endpoint_id, page=page, limit=limit)
    except ClientException as e:
        logger.exception(f"Failed to get deployment cache metrics: {e}")
        return ErrorResponse(code=e.status_code, message=e.message)
    except Exception as e:
        logger.exception(f"Failed to get deployment cache metrics: {e}")
        response = ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get deployment cache metrics"
        )
    return response.to_http_response()

@metric_router.post(
    "/analytics/inference-quality/{endpoint_id}",
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
            "model": InferenceQualityAnalyticsPromptResponse,
            "description": "Successfully get inference quality score analytics",
        },
    },
    description="Get inference quality score analytics",
)
async def get_inference_quality_score_analytics(
    endpoint_id: UUID,
    _: Annotated[User, Depends(get_current_active_user)],
    # session: Annotated[Session, Depends(get_session)],
) -> Union[InferenceQualityAnalyticsResponse, ErrorResponse]:
    """Get inference quality score analytics."""
    try:
        response = await BudMetricService().get_inference_quality_analytics(endpoint_id)
    except ClientException as e:
        logger.exception(f"Failed to get inference quality score analytics: {e}")
        response = ErrorResponse(code=e.status_code, message=e.message)
    except Exception as e:
        logger.exception(f"Failed to get inference quality score analytics: {e}")
        response = ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get inference quality score analytics"
        )
    return response.to_http_response()

@metric_router.post(
    "/analytics/inference-quality-prompts/{endpoint_id}/{score_type}",
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
            "model": InferenceQualityAnalyticsPromptResponse,
            "description": "Successfully get inference quality prompt analytics",
        },
    },
    description="Get inference quality prompt analytics",
)
async def get_inference_quality_prompt_analytics(
    endpoint_id: UUID,
    score_type: Literal["hallucination", "harmfulness", "sensitive_info", "prompt_injection"],
    _: Annotated[User, Depends(get_current_active_user)],
    # session: Annotated[Session, Depends(get_session)],
    filters: Annotated[InferenceQualityAnalyticsPromptFilter, Depends()],
    search: bool = False,
    order_by: Optional[List[str]] = Depends(parse_ordering_fields),
    page: int = 1,
    limit: int = 10,
) -> Union[InferenceQualityAnalyticsPromptResponse, ErrorResponse]:
    """Get inference quality prompt analytics."""
    try:
        order_by_str = ",".join(":".join(item) for item in order_by)
        response = await BudMetricService().get_inference_quality_prompt_analytics(endpoint_id, score_type, page, limit, filters, search, order_by_str if order_by_str else "created_at:desc")
    except ClientException as e:
        logger.exception(f"Failed to get inference quality prompt analytics: {e}")
        response = ErrorResponse(code=e.status_code, message=e.message)
    except Exception as e:
        logger.exception(f"Failed to get inference quality prompt analytics: {e}")
        response = ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR, message="Failed to get inference quality prompt analytics"
        )
    return response.to_http_response()
