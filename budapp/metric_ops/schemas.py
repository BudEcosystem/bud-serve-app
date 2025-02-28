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


"""Contains core Pydantic schemas used for data validation and serialization within the metric ops services."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import UUID4, BaseModel, ConfigDict, Field, model_validator

from ..commons.schemas import PaginatedSuccessResponse, SuccessResponse


class BaseAnalyticsRequest(BaseModel):
    """Base analytics request schema."""

    frequency: Literal["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]
    filter_by: Literal["project", "model", "endpoint"]
    filter_conditions: list[UUID4] | None = None
    project_id: UUID4 | None = None
    model_id: UUID4 | None = None
    from_date: datetime
    to_date: datetime | None = None
    top_k: int | None = None


class CountAnalyticsRequest(BaseAnalyticsRequest):
    """Request count analytics request schema."""

    metrics: Literal["global", "overall", "concurrency", "queuing_time", "input_output_tokens"] | None = None

    @model_validator(mode="before")
    def validate_filter_by(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that global metrics don't have filter conditions."""
        if data.get("metrics") == "global" and data.get("filter_conditions"):
            raise ValueError("global metrics doesn't support filter_conditions, use overall metrics instead")
        return data


class CountAnalyticsResponse(SuccessResponse):
    """Request count analytics response schema."""

    overall_metrics: dict | None = None
    concurrency_metrics: dict | None = None
    queuing_time_metrics: dict | None = None
    global_metrics: dict | None = None
    input_output_tokens_metrics: dict | None = None


class PerformanceAnalyticsRequest(BaseAnalyticsRequest):
    """Request performance analytics request schema."""

    metrics: Literal["ttft", "latency", "throughput"] | None = None


class PerformanceAnalyticsResponse(SuccessResponse):
    """Request performance analytics response schema."""

    ttft_metrics: dict | None = None
    latency_metrics: dict | None = None
    throughput_metrics: dict | None = None


class DashboardStatsResponse(SuccessResponse):
    """Dashboard stats response schema."""

    total_model_count: int
    cloud_model_count: int
    local_model_count: int
    total_projects: int
    total_project_users: int
    total_endpoints_count: int
    running_endpoints_count: int
    total_clusters: int
    inactive_clusters: int


class CacheMetricsResponse(SuccessResponse):
    object: str = "cache_metrics"
    latency: Optional[float] = None
    hit_ratio: Optional[float] = None
    most_reused_prompts: Optional[List[tuple[str, int]]] = None


class InferenceQualityAnalyticsResponse(SuccessResponse):
    object: str = "inference_quality_analytics"
    hallucination_score: Optional[float] = None
    harmfulness_score: Optional[float] = None
    sensitive_info_score: Optional[float] = None
    prompt_injection_score: Optional[float] = None


class InferenceQualityAnalyticsPromptResult(BaseModel):
    request_id: UUID
    prompt: str
    response: str
    score: float
    created_at: datetime


class InferenceQualityAnalyticsPromptResponse(PaginatedSuccessResponse):
    """Inference quality analytics prompt response schema."""

    model_config = ConfigDict(extra="allow")

    object: str = "inference_quality_analytics_prompt"
    score_type: Literal["hallucination", "harmfulness", "sensitive_info", "prompt_injection"]
    items: List[InferenceQualityAnalyticsPromptResult]
    total_items: int
    total_record: int = Field(..., alias="total_items")


class InferenceQualityAnalyticsPromptFilter(BaseModel):
    min_score: float = 0.0
    max_score: float = 1.0
    created_at: datetime | None = None
    prompt: str | None = None
    response: str | None = None
