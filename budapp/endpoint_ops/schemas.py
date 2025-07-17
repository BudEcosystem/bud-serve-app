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
from typing import Any, Optional, Union
from uuid import UUID

from pydantic import UUID4, BaseModel, ConfigDict, Field, model_validator

from budapp.cluster_ops.schemas import ClusterResponse
from budapp.commons.constants import AdapterStatusEnum, EndpointStatusEnum, ModelEndpointEnum, ProxyProviderEnum
from budapp.commons.schemas import PaginatedSuccessResponse, SuccessResponse
from budapp.model_ops.schemas import ModelDetailResponse, ModelResponse


# Endpoint schemas


class EndpointCreate(BaseModel):
    """Create endpoint schema."""

    project_id: UUID4
    model_id: UUID4
    cluster_id: Optional[UUID4] = None
    bud_cluster_id: Optional[UUID4] = None
    name: str
    url: str
    namespace: str
    status: EndpointStatusEnum
    created_by: UUID4
    status_sync_at: datetime
    credential_id: UUID4 | None
    active_replicas: int
    total_replicas: int
    number_of_nodes: int
    deployment_config: dict | None
    node_list: list | None
    supported_endpoints: list[ModelEndpointEnum]


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
    cluster: Optional[ClusterResponse] = None
    created_at: datetime
    modified_at: datetime
    is_deprecated: bool
    supported_endpoints: list[ModelEndpointEnum]


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
    device_name: str
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
    concurrency: int


class WorkerInfo(WorkerData):
    """Worker info."""

    model_config = ConfigDict(orm_mode=True, from_attributes=True)

    id: UUID


class WorkerInfoResponse(PaginatedSuccessResponse):
    """Response body for getting worker info."""

    model_config = ConfigDict(extra="allow")

    workers: list[WorkerInfo]


class WorkerLogsResponse(SuccessResponse):
    """Worker logs response."""

    model_config = ConfigDict(extra="allow")

    logs: Any


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
    cluster: Optional[ClusterResponse] = None
    deployment_config: Optional[dict] = None
    running_worker_count: int | None = None
    crashed_worker_count: int | None = None


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

    @model_validator(mode="after")
    def validate_fields(self) -> "AddWorkerRequest":
        """Validate the fields of the request."""
        if self.workflow_id is None and self.workflow_total_steps is None:
            raise ValueError("workflow_total_steps is required when workflow_id is not provided")

        if self.workflow_id is not None and self.workflow_total_steps is not None:
            raise ValueError("workflow_total_steps and workflow_id cannot be provided together")

        return self


class AddWorkerWorkflowStepData(BaseModel):
    """Add worker workflow step data."""

    endpoint_id: UUID4 | None = None
    additional_concurrency: int | None = None


class DeleteWorkerRequest(BaseModel):
    """Delete worker request."""

    endpoint_id: UUID4
    worker_id: UUID4
    worker_name: str


class WorkerMetricsResponse(SuccessResponse):
    """Worker metrics response."""

    model_config = ConfigDict(extra="allow")

    metrics: Union[dict[str, Any], None] = None


class AddAdapterRequest(BaseModel):
    workflow_id: UUID4 | None = None
    workflow_total_steps: int | None = None
    step_number: int = Field(..., gt=0)
    trigger_workflow: bool = False
    endpoint_id: UUID4 | None = None
    adapter_name: str | None = None
    adapter_model_id: UUID4 | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "AddAdapterRequest":
        """Validate the fields of the request."""
        if self.workflow_id is None and self.workflow_total_steps is None:
            raise ValueError("workflow_total_steps is required when workflow_id is not provided")

        if self.workflow_id is not None and self.workflow_total_steps is not None:
            raise ValueError("workflow_total_steps and workflow_id cannot be provided together")

        return self


class AddAdapterWorkflowStepData(BaseModel):
    """Add adapter workflow step data."""

    endpoint_id: UUID4 | None = None
    project_id: UUID4 | None = None
    adapter_name: str | None = None
    adapter_model_id: UUID4 | None = None
    adapter_id: UUID4 | None = None


class AdapterFilter(BaseModel):
    """Adapter filter."""

    name: str | None = None
    status: AdapterStatusEnum | None = None


class AdapterResponse(BaseModel):
    """Adapter response."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID4
    name: str
    status: AdapterStatusEnum
    model: ModelResponse
    created_at: datetime


class AdapterPaginatedResponse(PaginatedSuccessResponse):
    """Adapter paginated response."""

    adapters: list[AdapterResponse] = []


class VLLMConfig(BaseModel):
    """VLLM config."""

    type: str
    model_name: str
    api_base: str
    api_key_location: str


class ProxyModelConfig(BaseModel):
    """Proxy model config."""

    routing: list[ProxyProviderEnum]
    providers: dict[ProxyProviderEnum, VLLMConfig]
    endpoints: list[str]
