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
from typing import Literal

from pydantic import UUID4, BaseModel

from ..commons.schemas import SuccessResponse


class BaseAnalyticsRequest(BaseModel):
    """Base analytics request schema."""

    frequency: Literal["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]
    filter_by: Literal["project", "model", "endpoint"]
    filter_conditions: list[UUID4] | None = None
    from_date: datetime
    to_date: datetime | None = None
    top_k: int | None = None


class CountAnalyticsRequest(BaseAnalyticsRequest):
    """Request count analytics request schema."""

    metrics: Literal["overall", "concurrency"] = "overall"


class CountAnalyticsResponse(SuccessResponse):
    """Request count analytics response schema."""

    overall_metrics: dict
    concurrency_metrics: dict | None = None


class PerformanceAnalyticsRequest(BaseAnalyticsRequest):
    """Request performance analytics request schema."""

    metrics: Literal["ttft", "latency", "throughput"] = "ttft"


class PerformanceAnalyticsResponse(SuccessResponse):
    """Request performance analytics response schema."""

    data: str  # TODO: Remove this once we have the actual data
