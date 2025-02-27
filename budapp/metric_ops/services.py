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
from uuid import UUID

import aiohttp
from fastapi import status

from budapp.commons import logging
from budapp.commons.config import app_settings
from budapp.commons.db_utils import SessionMixin
from budapp.commons.exceptions import ClientException

from ..cluster_ops.crud import ClusterDataManager
from ..cluster_ops.models import Cluster as ClusterModel
from ..commons.constants import (
    ClusterStatusEnum,
    EndpointStatusEnum,
    ModelProviderTypeEnum,
    ModelStatusEnum,
    ProjectStatusEnum,
)
from ..endpoint_ops.crud import EndpointDataManager
from ..endpoint_ops.models import Endpoint as EndpointModel
from ..model_ops.crud import ModelDataManager
from ..model_ops.models import Model
from ..project_ops.crud import ProjectDataManager
from ..project_ops.models import Project as ProjectModel
from .schemas import (
    CacheMetricsResponse,
    CountAnalyticsRequest,
    CountAnalyticsResponse,
    DashboardStatsResponse,
    InferenceQualityAnalyticsPromptResponse,
    InferenceQualityAnalyticsResponse,
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
            queuing_time_metrics=bud_metric_response["queuing_time_metrics"],
            global_metrics=bud_metric_response["global_metrics"],
            input_output_tokens_metrics=bud_metric_response["input_output_tokens_metrics"],
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

    async def get_dashboard_stats(self, user_id: UUID) -> DashboardStatsResponse:
        """Fetch dashboard statistics for the given user."""
        db_total_model_count = await ModelDataManager(self.session).get_count_by_fields(
            Model, fields={"status": ModelStatusEnum.ACTIVE}
        )
        db_cloud_model_count = await ModelDataManager(self.session).get_count_by_fields(
            Model,
            fields={
                "status": ModelStatusEnum.ACTIVE,
                "provider_type": ModelProviderTypeEnum.CLOUD_MODEL,
            },
        )
        db_local_model_count = await ModelDataManager(self.session).get_count_by_fields(
            Model,
            fields={"status": ModelStatusEnum.ACTIVE},
            exclude_fields={"provider_type": ModelProviderTypeEnum.CLOUD_MODEL},
        )
        db_total_endpoint_count = await EndpointDataManager(self.session).get_count_by_fields(
            EndpointModel, fields={}, exclude_fields={"status": EndpointStatusEnum.DELETED}
        )
        db_running_endpoint_count = await EndpointDataManager(self.session).get_count_by_fields(
            EndpointModel, fields={"status": EndpointStatusEnum.RUNNING}
        )

        db_total_clusters = await ClusterDataManager(self.session).get_count_by_fields(
            ClusterModel, fields={}, exclude_fields={"status": ClusterStatusEnum.DELETED}
        )

        _, db_inactive_clusters = await ClusterDataManager(self.session).get_inactive_clusters()

        db_project_count = await ProjectDataManager(self.session).get_count_by_fields(
            ProjectModel, fields={"status": ProjectStatusEnum.ACTIVE}
        )

        db_total_project_users = ProjectDataManager(self.session).get_unique_user_count_in_all_projects()

        db_dashboard_stats = {
            "total_model_count": db_total_model_count,
            "cloud_model_count": db_cloud_model_count,
            "local_model_count": db_local_model_count,
            "total_projects": db_project_count,
            "total_project_users": db_total_project_users,
            "total_endpoints_count": db_total_endpoint_count,
            "running_endpoints_count": db_running_endpoint_count,
            "total_clusters": db_total_clusters,
            "inactive_clusters": db_inactive_clusters,
        }

        db_dashboard_stats = DashboardStatsResponse(
            code=status.HTTP_200_OK,
            object="dashboard.count",
            message="Successfully fetched dashboard count statistics",
            **db_dashboard_stats,
        )

        return db_dashboard_stats

    @staticmethod
    async def _perform_deployment_cache_metric(endpoint_id: UUID) -> Dict:
        """Get deployment cache metrics."""
        deployment_cache_metric_endpoint = f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_metrics_app_id}/method/metrics/analytics/cache-metrics/{endpoint_id}"

        logger.debug(
            f"Performing request deployment cache-metrics request to bud_metric for endpoint id : {endpoint_id}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(deployment_cache_metric_endpoint) as response:
                    response_data = await response.json()
                    if response.status != status.HTTP_200_OK:
                        if response.status == status.HTTP_404_NOT_FOUND:
                            response_data = {
                                "latency": None,
                                "hit_ratio": None,
                                "most_reused_prompts": [],
                            }
                        else:
                            logger.error(f"Failed to get deployment cache metrics: {response.status} {response_data}")
                            raise ClientException(
                                "Failed to get deployment cache metrics", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                            )

                    logger.debug("Successfully get deployment cache metrics from budmetric")
                    return response_data
        except Exception as e:
            logger.exception(f"Failed to send deployment cache metrics request: {e}")
            raise ClientException(
                "Failed to get deployment cache metrics", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from e


    async def get_deployment_cache_metric(self, endpoint_id: UUID) -> CacheMetricsResponse:
        """Get deployment cache metrics."""
        bud_metric_response = await self._perform_deployment_cache_metric(endpoint_id)

        return CacheMetricsResponse(
            code=status.HTTP_200_OK,
            object="deployment.cache.metrics",
            message="Successfully fetched deployment cache metrics",
            latency=bud_metric_response["latency"],
            hit_ratio=bud_metric_response["hit_ratio"],
            most_reused_prompts=bud_metric_response["most_reused_prompts"],
        )

    @staticmethod
    async def _perform_inference_quality_analytics(endpoint_id: UUID) -> Dict:
        """Get inference quality analytics."""
        inference_quality_analytics_endpoint = f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_metrics_app_id}/method/metrics/analytics/inference-quality/{endpoint_id}"

        logger.debug(
            f"Performing request inference quality analytics request to bud_metric for endpoint id : {endpoint_id}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(inference_quality_analytics_endpoint) as response:
                    response_data = await response.json()
                    if response.status != status.HTTP_200_OK:
                        logger.error(f"Failed to get inference quality analytics: {response.status} {response_data}")
                        raise ClientException(
                            "Failed to get inference quality analytics", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                        )

                    logger.debug("Successfully get inference quality analytics from budmetric")
                    return response_data
        except Exception as e:
            logger.exception(f"Failed to send inference quality analytics request: {e}")
            raise ClientException(
                "Failed to get inference quality analytics", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from e

    @staticmethod
    async def _perform_inference_quality_prompt_analytics(endpoint_id: UUID, score_type: str, page: int = 1, limit: int = 10, order_by: str = "created_at:desc") -> Dict:
        """Get inference quality prompt analytics."""
        inference_quality_prompt_analytics_endpoint = f"{app_settings.dapr_base_url}/v1.0/invoke/{app_settings.bud_metrics_app_id}/method/metrics/analytics/inference-quality/{score_type}/{endpoint_id}"

        logger.debug(
            f"Performing request inference quality prompt analytics request to bud_metric for endpoint id : {endpoint_id}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    inference_quality_prompt_analytics_endpoint,
                    params={
                        "page": page,
                        "limit": limit,
                        "order_by": order_by,
                    }
                ) as response:
                    response_data = await response.json()
                    if response.status != status.HTTP_200_OK:
                        logger.error(f"Failed to get inference quality prompt analytics: {response.status} {response_data}")
                        raise ClientException(
                            "Failed to get inference quality prompt analytics", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
                        )

                    logger.debug("Successfully get inference quality prompt analytics from budmetric")
                    return response_data
        except Exception as e:
            logger.exception(f"Failed to send inference quality prompt analytics request: {e}")
            raise ClientException(
                "Failed to get inference quality prompt analytics", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from e

    async def get_inference_quality_analytics(self, endpoint_id: UUID) -> InferenceQualityAnalyticsResponse:
        """Get inference quality analytics."""
        bud_metric_response = await self._perform_inference_quality_analytics(endpoint_id)

        return InferenceQualityAnalyticsResponse(
            code=status.HTTP_200_OK,
            object="inference.quality.analytics",
            message="Successfully fetched inference quality analytics",
            hallucination_score=bud_metric_response["hallucination_score"],
            harmfulness_score=bud_metric_response["harmfulness_score"],
            sensitive_info_score=bud_metric_response["sensitive_info_score"],
            prompt_injection_score=bud_metric_response["prompt_injection_score"],
        )

    async def get_inference_quality_prompt_analytics(self, endpoint_id: UUID, score_type: str, page: int = 1, limit: int = 10, order_by: str = "created_at:desc") -> InferenceQualityAnalyticsPromptResponse:
        """Get inference quality prompt analytics."""
        bud_metric_response = await self._perform_inference_quality_prompt_analytics(endpoint_id, score_type, page, limit, order_by)

        return InferenceQualityAnalyticsPromptResponse(
            code=status.HTTP_200_OK,
            message="Successfully fetched inference quality prompt analytics",
            **bud_metric_response,
        )
