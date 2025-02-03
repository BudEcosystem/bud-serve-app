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

from datetime import datetime
from typing import List, Dict, Union, Optional
from uuid import UUID
from enum import Enum

from pydantic import UUID4, AnyHttpUrl, BaseModel, ConfigDict, Field, computed_field, field_validator

from budapp.commons.constants import ClusterStatusEnum, EndpointStatusEnum
from budapp.commons.schemas import PaginatedSuccessResponse, SuccessResponse

from ..commons.helpers import validate_icon
from ..project_ops.schemas import Project
from ..model_ops.schemas import Model


class ClusterBase(BaseModel):
    """Cluster base schema."""

    name: str
    ingress_url: str
    icon: str


class ClusterCreate(ClusterBase):
    """Cluster create schema."""

    status: ClusterStatusEnum
    cpu_count: int
    gpu_count: int
    hpu_count: int
    cpu_total_workers: int
    cpu_available_workers: int
    gpu_total_workers: int
    gpu_available_workers: int
    hpu_total_workers: int
    hpu_available_workers: int
    created_by: UUID4
    cluster_id: UUID4
    status_sync_at: datetime


class ClusterResourcesInfo(BaseModel):
    """Cluster resources schema."""

    cpu_count: int
    gpu_count: int
    hpu_count: int
    cpu_total_workers: int
    cpu_available_workers: int
    gpu_total_workers: int
    gpu_available_workers: int
    hpu_total_workers: int
    hpu_available_workers: int


class ClusterResponse(BaseModel):
    """Cluster response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    icon: str
    ingress_url: str
    created_at: datetime
    modified_at: datetime
    status: ClusterStatusEnum
    cluster_id: UUID
    cpu_count: int
    gpu_count: int
    hpu_count: int
    cpu_total_workers: int
    cpu_available_workers: int
    gpu_total_workers: int
    gpu_available_workers: int
    hpu_total_workers: int
    hpu_available_workers: int

    @computed_field
    @property
    def total_nodes(self) -> int:
        """Total nodes."""
        return self.cpu_total_workers + self.gpu_total_workers + self.hpu_total_workers

    @computed_field
    @property
    def available_nodes(self) -> int:
        """Available nodes."""
        return self.cpu_available_workers + self.gpu_available_workers + self.hpu_available_workers


class ClusterPaginatedResponse(ClusterResponse):
    endpoint_count: int


class ClusterFilter(BaseModel):
    """Filter cluster schema."""

    name: str | None = None


class ClusterListResponse(PaginatedSuccessResponse):
    """Cluster response schema."""

    model_config = ConfigDict(extra="ignore")

    clusters: List[ClusterPaginatedResponse]


class CreateClusterWorkflowRequest(BaseModel):
    """Create cluster workflow request schema."""

    name: str | None = None
    icon: str | None = None
    ingress_url: str | None = None
    workflow_id: UUID4 | None = None
    workflow_total_steps: int | None = None
    step_number: int | None = None
    trigger_workflow: bool | None = None

    @field_validator("icon", mode="before")
    @classmethod
    def icon_validate(cls, value: str | None) -> str | None:
        """Validate the icon."""
        if value is not None and not validate_icon(value):
            raise ValueError("invalid icon")
        return value


class CreateClusterWorkflowSteps(BaseModel):
    """Create cluster workflow step data schema."""

    name: str | None = None
    icon: str | None = None
    ingress_url: AnyHttpUrl | None = None
    configuration_yaml: dict | None = None


class EditClusterRequest(BaseModel):
    name: str | None = Field(
        None,
        min_length=1,
        max_length=100,
        description="Name of the cluster, must be non-empty and at most 100 characters.",
    )
    icon: str | None = Field(None, description="URL or path of the cluster icon.")
    ingress_url: AnyHttpUrl | None = Field(None, description="ingress_url.")

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        """Ensure the name is not empty or only whitespace."""
        if value is not None and not value.strip():
            raise ValueError("Cluster name cannot be empty or only whitespace.")
        return value

    @field_validator("ingress_url", mode="after")
    @classmethod
    def convert_url_to_string(cls, value: AnyHttpUrl | None) -> str | None:
        """Convert AnyHttpUrl to string."""
        return str(value) if value is not None else None

    @field_validator("icon", mode="before")
    @classmethod
    def icon_validate(cls, value: str | None) -> str | None:
        """Validate the icon."""
        if value is not None and not validate_icon(value):
            raise ValueError("invalid icon")
        return value


class ClusterDetailResponse(ClusterResponse):
    """Cluster detail response schema"""

    total_workers_count: int
    active_workers_count: int
    total_endpoints_count: int
    running_endpoints_count: int
    hardware_type: list


class SingleClusterResponse(SuccessResponse):
    """Single cluster entity"""

    cluster: Union[ClusterResponse, ClusterDetailResponse]


class CancelClusterOnboardingRequest(BaseModel):
    """Cancel cluster onboarding request schema."""

    workflow_id: UUID4


class ClusterEndpointResponse(BaseModel):
    """Cluster endpoint response schema."""

    name: str
    status: EndpointStatusEnum
    created_at: datetime
    project: Project
    model: Model
    active_workers: int
    total_workers: int
    roi: int


class ClusterEndpointPaginatedResponse(PaginatedSuccessResponse):
    """Cluster endpoint paginated response schema."""

    endpoints: list[ClusterEndpointResponse] = []


class ClusterEndpointFilter(BaseModel):
    """Filter schema for endpoints."""

    name: str | None = None
    status: EndpointStatusEnum | None = None


# Cluster Metrics Schema
class TimeSeriesPoint(BaseModel):
    """A single point in a time series."""

    timestamp: int
    value: float


class NetworkInMetrics(BaseModel):
    """Network inbound metrics."""
    inbound_mbps: float
    change_percent: float
    time_series: Optional[List[TimeSeriesPoint]]


class NetworkOutMetrics(BaseModel):
    """Network outbound metrics."""
    outbound_mbps: float
    change_percent: float
    time_series: Optional[List[TimeSeriesPoint]]


class NetworkBandwidthMetrics(BaseModel):
    """Network total bandwidth metrics."""
    total_mbps: float
    change_percent: float
    time_series: Optional[List[TimeSeriesPoint]]


class ResourceMetrics(BaseModel):
    """Base metrics for resources."""

    total_gib: float
    used_gib: float
    available_gib: float
    usage_percent: float
    change_percent: float


class CPUMetrics(BaseModel):
    """CPU metrics."""

    usage_percent: float
    change_percent: float


class NodeMetrics(BaseModel):
    """Metrics for a single node."""
    memory: ResourceMetrics
    storage: ResourceMetrics
    cpu: CPUMetrics
    network_in: NetworkInMetrics
    network_out: NetworkOutMetrics
    network_bandwidth: NetworkBandwidthMetrics


class ClusterSummaryMetrics(BaseModel):
    """Summary metrics for the entire cluster."""
    total_nodes: int = 0  # Added default value
    memory: ResourceMetrics
    storage: ResourceMetrics
    cpu: CPUMetrics
    gpu: Optional[Dict[str, float]] = None  # Added default None
    hpu: Optional[Dict[str, float]] = None  # Added default None
    network_in: NetworkInMetrics
    network_out: NetworkOutMetrics
    network_bandwidth: NetworkBandwidthMetrics
    timestamp: str
    time_range: str
    cluster_id: str
    metric_type: str


class ClusterMetrics(BaseModel):
    """Complete cluster metrics."""
    nodes: Dict[str, NodeMetrics]
    cluster_summary: ClusterSummaryMetrics
    timestamp: str
    time_range: str
    cluster_id: str
    metric_type: str


class MetricTypeEnum(Enum):
    """Enum for metric types."""

    ALL = "all"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    GPU = "gpu"
    HPU = "hpu"
    NETWORK_IN = "network_in"
    NETWORK_OUT = "network_out"
    NETWORK_BANDWIDTH = "network_bandwidth"


class ClusterMetricsResponse(SuccessResponse):
    """Cluster metrics response schema."""
    nodes: any
    cluster_summary: any
    time_range: str
    metric_type: str
    timestamp: str
    cluster_id: str
