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

from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
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


class AddCloudModelWorkflowRequest(BaseModel):
    """Cloud model workflow request schema."""

    workflow_id: UUID4 | None = None
    workflow_total_steps: int | None = None
    step_number: int = Field(..., gt=0)
    trigger_workflow: bool = False
    provider_type: ModelProviderTypeEnum | None = None
    source: CredentialTypeEnum | None = None
    name: str | None = None
    modality: ModalityEnum | None = None
    uri: str | None = None
    tags: list[Tag] | None = None
    icon: str | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "AddCloudModelWorkflowRequest":
        """Validate the fields of the request."""
        if self.workflow_id is None and self.workflow_total_steps is None:
            raise ValueError("workflow_total_steps is required when workflow_id is not provided")

        if self.workflow_id is not None and self.workflow_total_steps is not None:
            raise ValueError("workflow_total_steps and workflow_id cannot be provided together")

        # Check if at least one of the other fields is provided
        other_fields = [
            self.provider_type,
            self.source,
            self.modality,
            self.uri,
            self.tags,
        ]
        if not any(other_fields):
            raise ValueError(
                "At least one of provider_type, source, modality, uri, or tags is required when workflow_id is provided"
            )

        return self


class AddCloudModelWorkflowStepData(BaseModel):
    """Cloud model workflow step data schema."""

    provider_type: ModelProviderTypeEnum | None = None
    source: CredentialTypeEnum | None = None
    name: str | None = None
    modality: ModalityEnum | None = None
    uri: str | None = None
    tags: list[Tag] | None = None
    icon: str | None = None


class AddCloudModelWorkflowStepDataResponse(BaseModel):
    """Cloud model workflow step data schema."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    provider_type: ModelProviderTypeEnum | None = None
    source: CredentialTypeEnum | None = None
    name: str | None = None
    modality: ModalityEnum | None = None
    uri: str | None = None
    tags: list[Tag] | None = None
    icon: str | None = None


class AddCloudModelWorkflowResponse(SuccessResponse):
    """Add Cloud Model Workflow Response."""

    workflow_id: UUID4
    status: WorkflowStatusEnum
    current_step: int
    total_steps: int
    reason: str | None = None
    workflow_steps: AddCloudModelWorkflowStepDataResponse | None = None
