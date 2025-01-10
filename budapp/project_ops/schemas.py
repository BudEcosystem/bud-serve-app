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


"""Contains core Pydantic schemas used for data validation and serialization within the project ops services."""

from datetime import datetime
from typing import List, Literal

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from budapp.commons.schemas import PaginatedSuccessResponse, SuccessResponse, Tag

from ..commons.constants import ClusterStatusEnum
from ..commons.helpers import validate_icon


class ProjectBase(BaseModel):
    name: str
    description: str | None = None
    tags: List[Tag] | None = None
    icon: str | None = None


class EditProjectRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = Field(None, max_length=400)
    tags: List[Tag] | None = None
    icon: str | None = None

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        """Ensure the name is not empty or only whitespace."""
        if value is not None and not value.strip():
            raise ValueError("Project name cannot be empty or only whitespace.")
        return value

    @field_validator("icon", mode="before")
    @classmethod
    def icon_validate(cls, value: str | None) -> str | None:
        """Validate the icon."""
        if value is not None and not validate_icon(value):
            raise ValueError("invalid icon")
        return value


class ProjectResponse(ProjectBase):
    """Project response to client schema"""

    model_config = ConfigDict(from_attributes=True)

    id: UUID4


class SingleProjectResponse(SuccessResponse):
    project: ProjectResponse


class ProjectClusterListResponse(BaseModel):
    """Project cluster list response schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID4
    name: str
    endpoint_count: int
    hardware_type: list[Literal["CPU", "GPU", "HPU"]]
    node_count: int
    worker_count: int
    status: ClusterStatusEnum
    created_at: datetime
    modified_at: datetime


class ProjectClusterPaginatedResponse(PaginatedSuccessResponse):
    """Project cluster paginated response schema."""

    clusters: list[ProjectClusterListResponse] = []


class ProjectClusterFilter(BaseModel):
    """Filter project cluster schema for filtering clusters based on specific criteria."""

    name: str | None = None
    status: ClusterStatusEnum | None = None
