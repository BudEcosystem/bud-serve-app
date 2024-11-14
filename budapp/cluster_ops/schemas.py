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


"""Contains core Pydantic schemas used for data validation and serialization within the cluster ops services."""

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
)
from uuid import UUID
from typing import List, Dict
from datetime import datetime

from budapp.commons.schemas import PaginatedSuccessResponse
from budapp.commons.constants import (
    ClusterTypeEnum,
    ClusterStatusEnum
)

class ClusterResponse(BaseModel):
    """Cluster response schema"""

    id: UUID
    name: str
    icon: str
    created_at: datetime
    modified_at: datetime
    endpoint_count: int
    status: ClusterStatusEnum
    resources: Dict[str, int]
    cluster_id: UUID

    model_config = ConfigDict(from_attributes=True)


class ClusterFilter(BaseModel):
    """Filter cluster schema"""

    name: str | None = None
    type: ClusterTypeEnum | None = None
    total_workers: int | None = None
    available_workers: int | None = None
    

class ClusterListResponse(PaginatedSuccessResponse):
    """Cluster response schema."""

    model_config = ConfigDict(extra="ignore")

    clusters: List[ClusterResponse]
