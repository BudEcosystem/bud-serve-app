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

"""The metric ops services. Contains business logic for metric ops."""

from typing import Dict

import aiohttp
from fastapi import status

from budapp.commons import logging
from budapp.commons.config import app_settings
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException

from .schemas import (
    CountAnalyticsRequest,
    CountAnalyticsResponse,
    PerformanceAnalyticsRequest,
    PerformanceAnalyticsResponse,
)


logger = logging.get_logger(__name__)


class MetricService(SessionMixin):
    """Metric service."""

    async def get_request_count_analytics(
        self,
        request: CountAnalyticsRequest,
    ) -> CountAnalyticsResponse:
        """Get request count analytics."""
        bud_metric_response = await self._perform_request_count_analytics(request)

        return CountAnalyticsResponse(
            code=status.HTTP_200_OK,
            object="request.count.analytics",
            message="Successfully fetched request count analytics",
            overall_metrics=bud_metric_response["overall_metrics"],
            concurrency_metrics=bud_metric_response["concurrency_metrics"],
        )

    async def get_request_performance_analytics(
        self,
        request: PerformanceAnalyticsRequest,
    ) -> PerformanceAnalyticsResponse:
        """Get request performance analytics."""
        bud_metric_response = await self._perform_request_performance_analytics(request)

        return PerformanceAnalyticsResponse(
            code=status.HTTP_200_OK,
            object="request.performance.analytics",
            message="Successfully fetched request performance analytics",
            ttft_metrics=bud_metric_response["ttft_metrics"],
            latency_metrics=bud_metric_response["latency_metrics"],
            throughput_metrics=bud_metric_response["throughput_metrics"],
        )

    @staticmethod
    async def _perform_request_count_analytics(
        metric_request: CountAnalyticsRequest,
    ) -> Dict:
        """Get request count analytics."""
        request_count_analytics_endpoint = f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_metrics_app_id}/method/metrics/analytics/request-counts"

        logger.debug(
            f"Performing request count analytics request to bud_metric {metric_request.model_dump(exclude_none=True, exclude_unset=True, mode='json')}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    request_count_analytics_endpoint,
                    json=metric_request.model_dump(exclude_none=True, exclude_unset=True, mode="json"),
                ) as response:
                    response_data = await response.json()
                    if response.status != status.HTTP_200_OK:
                        logger.error(f"Failed to get request count analytics: {response.status} {response_data}")
                        raise ClientException(
                            "Failed to get request count analytics", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                        )

                    logger.debug("Successfully get request count analytics from budmetric")
                    return response_data
        except Exception as e:
            logger.exception(f"Failed to send request count analytics request: {e}")
            raise ClientException(
                "Failed to get request count analytics", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from e

    @staticmethod
    async def _perform_request_performance_analytics(
        metric_request: PerformanceAnalyticsRequest,
    ) -> Dict:
        """Get request performance analytics."""
        request_performance_analytics_endpoint = f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_metrics_app_id}/method/metrics/analytics/request-performance"

        logger.debug(
            f"Performing request performance analytics request to bud_metric {metric_request.model_dump(exclude_none=True, exclude_unset=True, mode='json')}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    request_performance_analytics_endpoint,
                    json=metric_request.model_dump(exclude_none=True, exclude_unset=True, mode="json"),
                ) as response:
                    response_data = await response.json()
                    if response.status != status.HTTP_200_OK:
                        logger.error(f"Failed to get request performance analytics: {response.status} {response_data}")
                        raise ClientException(
                            "Failed to get request performance analytics",
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        )

                    logger.debug("Successfully get request performance analytics from budmetric")
                    return response_data
        except Exception as e:
            logger.exception(f"Failed to send request performance analytics request: {e}")
            raise ClientException(
                "Failed to get request performance analytics", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from e
