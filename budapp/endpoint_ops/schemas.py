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


"""Contains core Pydantic schemas used for data validation and serialization within the core services."""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import UUID4, BaseModel, ConfigDict

from budapp.cluster_ops.schemas import ClusterResponse
from budapp.commons.constants import EndpointStatusEnum
from budapp.commons.schemas import PaginatedSuccessResponse
from budapp.model_ops.schemas import ModelResponse


# Endpoint schemas


class EndpointCreate(BaseModel):
    """Create endpoint schema."""

    project_id: UUID4
    model_id: UUID4
    cluster_id: UUID4
    bud_cluster_id: UUID4
    name: str
    url: str
    namespace: str
    status: EndpointStatusEnum
    created_by: UUID4
    status_sync_at: datetime
    credential_id: UUID4 | None


class EndpointFilter(BaseModel):
    """Filter endpoint schema for filtering endpoints based on specific criteria."""

    name: str | None = None
    status: EndpointStatusEnum | None = None


class EndpointListResponse(BaseModel):
    """Endpoint list response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID4
    name: str
    status: EndpointStatusEnum
    model: ModelResponse
    cluster: ClusterResponse
    created_at: datetime
    modified_at: datetime


class EndpointPaginatedResponse(PaginatedSuccessResponse):
    """Endpoint paginated response schema."""

    endpoints: list[EndpointListResponse] = []
    

class WorkerInfoFilter(BaseModel):
    """Filter for worker info."""

    status: str | None = None
    hardware: str | None = None
    utilization_min: int | None = None
    utilization_max: int | None = None


class DeploymentStatusEnum(str, Enum):
    READY = "ready"
    PENDING = "pending"
    INGRESS_FAILED = "ingress_failed"
    FAILED = "failed"


class WorkerData(BaseModel):
    """Worker data."""

    cluster_id: Optional[UUID] = None
    namespace: Optional[str] = None
    name: str
    status: str
    node_name: str
    utilization: Optional[str] = None
    hardware: str
    uptime: str
    last_restart_date: Optional[str] = None
    last_updated_date: Optional[str] = None
    created_date: str
    node_ip: str
    cores: int
    memory: str
    deployment_status: Optional[DeploymentStatusEnum] = None


class WorkerInfo(WorkerData):
    """Worker info."""

    model_config = ConfigDict(orm_mode=True, from_attributes=True)

    id: UUID
    

class WorkerInfoResponse(PaginatedSuccessResponse):
    """Response body for getting worker info."""
    
    model_config = ConfigDict(extra="allow")

    workers: list[WorkerInfo]
