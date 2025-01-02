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

from pydantic import UUID4, AnyHttpUrl, BaseModel, ConfigDict, Field, computed_field, field_validator

from budapp.commons.constants import ClusterStatusEnum
from budapp.commons.schemas import PaginatedSuccessResponse, SuccessResponse
from ..commons.helpers import validate_icon


def validate_icon_field(value: str | None) -> str | None:
    """Utility function to validate the icon field."""
    if value is not None and not validate_icon(value):
        raise ValueError("invalid icon")
    return value


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
        return validate_icon_field(value)


class SingleClusterResponse(SuccessResponse):
    cluster: ClusterResponse


class CancelClusterOnboardingRequest(BaseModel):
    """Cancel cluster onboarding request schema."""

    workflow_id: UUID4
