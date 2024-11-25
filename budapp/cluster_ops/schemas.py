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
from typing import List
from uuid import UUID

from pydantic import UUID4, AnyHttpUrl, BaseModel, ConfigDict, computed_field

from budapp.commons.constants import ClusterStatusEnum
from budapp.commons.schemas import PaginatedSuccessResponse


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
    created_at: datetime
    modified_at: datetime
    endpoint_count: int
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


class ClusterFilter(BaseModel):
    """Filter cluster schema."""

    name: str | None = None


class ClusterListResponse(PaginatedSuccessResponse):
    """Cluster response schema."""

    model_config = ConfigDict(extra="ignore")

    clusters: List[ClusterResponse]


class CreateClusterWorkflowRequest(BaseModel):
    """Create cluster workflow request schema."""

    name: str | None = None
    icon: str | None = None
    ingress_url: str | None = None
    workflow_id: UUID4 | None = None
    workflow_total_steps: int | None = None
    step_number: int | None = None
    trigger_workflow: bool | None = None


class CreateClusterWorkflowSteps(BaseModel):
    """Create cluster workflow step data schema."""

    name: str | None = None
    icon: str | None = None
    ingress_url: AnyHttpUrl | None = None
    configuration_yaml: dict | None = None


class EditCluster(BaseModel):
    name: str | None = None
    ingress_url: AnyHttpUrl | None = None
