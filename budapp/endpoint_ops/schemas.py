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

from pydantic import UUID4, BaseModel, ConfigDict, Field, model_validator

from budapp.cluster_ops.schemas import ClusterResponse
from budapp.commons.constants import EndpointStatusEnum
from budapp.commons.schemas import PaginatedSuccessResponse, SuccessResponse
from budapp.model_ops.schemas import ModelDetailResponse, ModelResponse


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
    total_replicas: int
    number_of_nodes: int
    deployment_config: dict | None


class EndpointFilter(BaseModel):
    """Filter endpoint schema for filtering endpoints based on specific criteria."""

    name: str | None = None
    status: EndpointStatusEnum | None = None


class EndpointResponse(BaseModel):
    """Endpoint response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID4
    name: str
    status: EndpointStatusEnum
    deployment_config: dict
    created_at: datetime
    modified_at: datetime


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
    last_restart_datetime: Optional[datetime] = None
    last_updated_datetime: Optional[datetime] = None
    created_datetime: datetime
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


class WorkerDetailResponse(SuccessResponse):
    """Worker detail response."""

    model_config = ConfigDict(extra="allow")

    worker: WorkerInfo


class ModelClusterDetail(BaseModel):
    """Model cluster detail."""

    model_config = ConfigDict(extra="allow")

    id: UUID
    name: str
    status: str
    model: ModelDetailResponse
    cluster: ClusterResponse


class ModelClusterDetailResponse(SuccessResponse):
    """Model cluster detail response."""

    model_config = ConfigDict(extra="allow")

    result: ModelClusterDetail


class AddWorkerRequest(BaseModel):
    """Add worker request."""

    workflow_id: UUID4 | None = None
    workflow_total_steps: int | None = None
    step_number: int = Field(..., gt=0)
    trigger_workflow: bool = False
    endpoint_id: UUID4 | None = None
    additional_concurrency: int | None = Field(None, gt=0)
    cluster_id: UUID4 | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "AddWorkerRequest":
        """Validate the fields of the request."""
        if self.workflow_id is None and self.workflow_total_steps is None:
            raise ValueError("workflow_total_steps is required when workflow_id is not provided")

        if self.workflow_id is not None and self.workflow_total_steps is not None:
            raise ValueError("workflow_total_steps and workflow_id cannot be provided together")

        # Check if at least one of the other fields is provided
        other_fields = [self.endpoint_id, self.additional_concurrency, self.cluster_id]
        required_fields = ["endpoint_id", "additional_concurrency", "cluster_id"]
        if not any(other_fields):
            raise ValueError(f"At least one of {', '.join(required_fields)} is required")

        return self


class AddWorkerWorkflowStepData(BaseModel):
    """Add worker workflow step data."""

    endpoint_id: UUID4 | None = None
    cluster_id: UUID4 | None = None
    additional_concurrency: int | None = None
