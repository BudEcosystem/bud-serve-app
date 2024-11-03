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


"""Contains core Pydantic schemas used for data validation and serialization within the model ops services."""

import re
from datetime import datetime
from typing import List, Optional, Tuple

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    validator,
)

from budapp.commons.constants import (
    CredentialTypeEnum,
    ModalityEnum,
    ModelProviderTypeEnum,
    WorkflowStatusEnum,
)
from budapp.commons.schemas import PaginatedSuccessResponse, SuccessResponse


class ProviderFilter(BaseModel):
    """Provider filter schema."""

    name: str | None = None


class Provider(BaseModel):
    """Provider schema."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID4
    name: str
    description: str
    type: CredentialTypeEnum
    icon: str


class ProviderResponse(PaginatedSuccessResponse):
    """Provider response schema."""

    model_config = ConfigDict(extra="ignore")

    providers: list[Provider] = []


class Tag(BaseModel):
    """Tag schema with name and color."""

    name: str = Field(..., min_length=1)
    color: str = Field(..., pattern="^#[0-9A-Fa-f]{6}$")

    @field_validator("color")
    def validate_hex_color(cls, v: str) -> str:
        """Validate that color is a valid hex color code."""
        if not re.match(r"^#[0-9A-Fa-f]{6}$", v):
            raise ValueError("Color must be a valid hex color code (e.g., #FF0000)")
        return v.upper()  # Normalize to uppercase


class Model(BaseModel):
    """Model schema."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID4
    name: str
    description: str | None = None
    modality: ModalityEnum
    source: CredentialTypeEnum
    provider_type: ModelProviderTypeEnum
    uri: str
    model_size: int | None = None
    tags: list[Tag] | None = None
    tasks: list[Tag] | None = None
    icon: str
    github_url: str | None = None
    huggingface_url: str | None = None
    website_url: str | None = None
    created_by: UUID4 | None = None
    author: str | None = None
    created_at: datetime
    modified_at: datetime


class CloudModel(BaseModel):
    """Cloud model schema."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID4
    name: str
    description: str | None = None
    modality: ModalityEnum
    source: CredentialTypeEnum
    provider_type: ModelProviderTypeEnum
    uri: str
    model_size: int | None = None
    tags: list[Tag] | None = None
    tasks: list[Tag] | None = None


class CreateCloudModelWorkflowRequest(BaseModel):
    """Cloud model workflow request schema."""

    workflow_id: UUID4 | None = None
    workflow_total_steps: int | None = None
    step_number: int = Field(..., gt=0)
    trigger_workflow: bool = False
    provider_type: ModelProviderTypeEnum | None = None
    provider_id: UUID4 | None = None
    name: str | None = None
    modality: ModalityEnum | None = None
    uri: str | None = None
    tags: list[Tag] | None = None
    cloud_model_id: UUID4 | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "CreateCloudModelWorkflowRequest":
        """Validate the fields of the request."""
        if self.workflow_id is None and self.workflow_total_steps is None:
            raise ValueError("workflow_total_steps is required when workflow_id is not provided")

        if self.workflow_id is not None and self.workflow_total_steps is not None:
            raise ValueError("workflow_total_steps and workflow_id cannot be provided together")

        # Check if at least one of the other fields is provided
        other_fields = [
            self.provider_type,
            self.provider_id,
            self.modality,
            self.uri,
            self.tags,
            self.name,
        ]
        required_fields = ["provider_type", "provider_id", "modality", "uri", "tags", "name"]
        if not any(other_fields):
            # Allow if cloud model id is explicitly provided
            input_data = self.model_dump(exclude_unset=True)
            if "cloud_model_id" in input_data:
                return self
            raise ValueError(f"At least one of {', '.join(required_fields)} is required when workflow_id is provided")

        return self


class EditModel(BaseModel):
    """Schema for editing a model with optional fields and validations."""

    name: Optional[str] = Field(None, min_length=1, max_length=100, description="Model name")
    description: Optional[str] = Field(None, max_length=500, description="Brief model description")
    modality: Optional[ModalityEnum] = None
    source: Optional[str] = None
    provider_type: Optional[ModelProviderTypeEnum] = None
    uri: Optional[str] = Field(None, description="Direct URI of the model")
    model_size: Optional[int] = Field(None, gt=0, description="Size of the model in bytes")
    tags: Optional[List[Tag]] = None
    tasks: Optional[List[Tag]] = None
    icon: Optional[str] = Field(None, description="URL for the model's icon")
    github_url: Optional[str] = Field(None, description="URL to the model's GitHub repository")
    huggingface_url: Optional[str] = Field(None, description="URL to the model's Hugging Face page")
    website_url: Optional[str] = Field(None, description="URL to the model's official website")
    created_by: Optional[UUID4] = Field(None, description="UUID of the user who created the model")
    author: Optional[str] = Field(None, max_length=100, description="Author name")

    @validator('name')
    def validate_name(cls, v):
        if v and not v.isalnum():
            raise ValueError("Model name must be alphanumeric")
        return v
    
    @validator('model_size')
    def validate_model_size(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Model size must be a positive integer")
        return v



class CreateCloudModelWorkflowSteps(BaseModel):
    """Cloud model workflow step data schema."""

    provider_type: ModelProviderTypeEnum | None = None
    source: str | None = None
    name: str | None = None
    modality: ModalityEnum | None = None
    uri: str | None = None
    tags: list[Tag] | None = None
    icon: str | None = None
    provider_id: UUID4 | None = None
    cloud_model_id: UUID4 | None = None


class CreateCloudModelWorkflowStepData(BaseModel):
    """Cloud model workflow step data schema."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    provider_type: ModelProviderTypeEnum | None = None
    provider: Provider | None = None
    cloud_model: CloudModel | None = None
    cloud_model_id: UUID4 | None = None
    provider_id: UUID4 | None = None
    model_id: UUID4 | None = None
    model: Model | None = None
    workflow_execution_status: dict | None = None
    leaderboard: list | None = None


class CreateCloudModelWorkflowResponse(SuccessResponse):
    """Add Cloud Model Workflow Response."""

    workflow_id: UUID4
    status: WorkflowStatusEnum
    current_step: int
    total_steps: int
    reason: str | None = None
    workflow_steps: CreateCloudModelWorkflowStepData | None = None


class CloudModelFilter(BaseModel):
    """Cloud model filter schema."""

    model_config = ConfigDict(protected_namespaces=())

    source: CredentialTypeEnum | None = None
    modality: ModalityEnum | None = None
    model_size: int | None = None
    name: str | None = None

    @field_validator("source")
    def change_to_string(cls, v: CredentialTypeEnum | None) -> str | None:
        """Change the source to a string."""
        return v.value if v else None


class CloudModelResponse(PaginatedSuccessResponse):
    """Cloud model response schema."""

    model_config = ConfigDict(extra="ignore")

    cloud_models: list[CloudModel] = []


class TagWithCount(BaseModel):
    """Tag with count schema."""

    name: str
    color: str = Field(..., pattern="^#[0-9A-Fa-f]{6}$")
    count: int

    @field_validator("color")
    def validate_hex_color(cls, v: str) -> str:
        """Validate that color is a valid hex color code."""
        if not re.match(r"^#[0-9A-Fa-f]{6}$", v):
            raise ValueError("Color must be a valid hex color code (e.g., #FF0000)")
        return v.upper()  # Normalize to uppercase


class RecommendedTagsResponse(PaginatedSuccessResponse):
    """Recommended tags response schema."""

    tags: List[TagWithCount] = []

    @field_validator("tags", mode="before")
    def validate_tags(cls, v: List[Tuple[str, str, int]]) -> List[TagWithCount]:
        """Convert tuples to TagWithCount objects."""
        return [TagWithCount(name=tag[0], color=tag[1], count=tag[2]) for tag in v]


class ModelCreate(BaseModel):
    """Schema for creating a new AI Model."""

    model_config = ConfigDict(protected_namespaces=())

    name: str
    description: str | None = None
    tags: List[Tag] | None = None
    tasks: List[Tag] | None = None
    author: str | None = None
    model_size: int | None = None
    icon: str
    github_url: str | None = None
    huggingface_url: str | None = None
    website_url: str | None = None
    modality: ModalityEnum
    source: str
    provider_type: ModelProviderTypeEnum
    uri: str
    created_by: UUID4
