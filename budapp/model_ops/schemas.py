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
from typing import List, Literal, Optional, Tuple

from fastapi import UploadFile
from pydantic import (
    UUID4,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_serializer,
    field_validator,
    model_validator,
)

from budapp.commons.constants import (
    BaseModelRelationEnum,
    CredentialTypeEnum,
    ModalityEnum,
    ModelProviderTypeEnum,
    ModelSecurityScanStatusEnum,
    ModelSourceEnum,
    WorkflowStatusEnum,
)
from budapp.commons.schemas import PaginatedSuccessResponse, SuccessResponse, Tag, Task
from budapp.user_ops.schemas import UserInfo


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
    is_present_in_model: bool = False


class PaperPublishedModel(BaseModel):
    """Paper Published Model Schema"""

    id: UUID4
    title: str | None = None
    authors: list[str] | None = None
    url: str | None = None
    model_id: UUID4

    class Config:
        orm_mode = True
        from_attributes = True


class PaperPublishedModelEditRequest(BaseModel):
    """Paper Published Edit Model Schema"""

    id: UUID4 | None = None
    title: str | None = None
    url: str


class ModelLicensesModel(BaseModel):
    """Model Licenses Model Schema"""

    id: UUID4
    name: str | None = None
    url: str | None = None
    path: str | None = None
    faqs: list[dict] | None = None
    model_id: UUID4

    class Config:
        orm_mode = True
        from_attributes = True


# Model related schemas


class ModelBase(BaseModel):
    """Base model schema."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    name: str
    description: Optional[str] = None
    tags: Optional[List[Tag]] = None
    tasks: Optional[List[Tag]] = None
    github_url: Optional[str] = None
    huggingface_url: Optional[str] = None
    website_url: Optional[str] = None


class Model(ModelBase):
    """Model schema."""

    id: UUID4
    icon: str | None = None
    modality: ModalityEnum
    source: ModelSourceEnum
    provider_type: ModelProviderTypeEnum
    uri: str
    model_size: Optional[int] = None
    created_by: Optional[UUID4] = None
    author: Optional[str] = None
    created_at: datetime
    modified_at: datetime
    provider: Provider | None = None


class ModelArchitecture(BaseModel):
    """Model architecture schema."""

    intermediate_size: int | None = None
    vocab_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    rope_scaling: dict | None = None
    model_weights_size: int | None = None
    kv_cache_size: int | None = None


class ModelCreate(ModelBase):
    """Schema for creating a new AI Model."""

    modality: ModalityEnum
    source: str
    provider_type: ModelProviderTypeEnum
    uri: str
    model_size: Optional[int] = None
    created_by: UUID4
    author: Optional[str] = None
    provider_id: UUID4 | None = None
    local_path: str | None = None
    strengths: list[str] | None = None
    limitations: list[str] | None = None
    languages: list[str] | None = None
    use_cases: list[str] | None = None
    minimum_requirements: dict | None = None
    examples: list[dict] | None = None
    base_model: str | None = None
    base_model_relation: BaseModelRelationEnum | None = None
    model_type: str | None = None
    family: str | None = None
    num_layers: int | None = None
    hidden_size: int | None = None
    context_length: int | None = None
    torch_dtype: str | None = None
    architecture: ModelArchitecture | None = None
    scan_verified: bool | None = None


class ModelDetailResponse(BaseModel):
    """Response schema for model details."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID4
    name: str
    description: str | None = None
    tags: list[Tag] | None = None
    tasks: list[Task] | None = None
    author: str | None = None
    model_size: int | None = None
    icon: str | None = None
    github_url: str | None = None
    huggingface_url: str | None = None
    website_url: str | None = None
    bud_verified: bool = False
    scan_verified: bool | None = None
    eval_verified: bool = False
    strengths: list[str] | None = None
    limitations: list[str] | None = None
    languages: list[str] | None = None
    use_cases: list[str] | None = None
    minimum_requirements: dict | None = None
    examples: list[dict] | None = None
    base_model: str | None = None
    model_type: str | None = None
    family: str | None = None
    num_layers: int | None = None
    hidden_size: int | None = None
    context_length: int | None = None
    torch_dtype: str | None = None
    architecture: ModelArchitecture | None = None
    modality: ModalityEnum
    source: str
    provider_type: ModelProviderTypeEnum
    uri: str
    paper_published: list[PaperPublishedModel] | None = None
    model_licenses: ModelLicensesModel | None = None
    provider: Provider | None = None


class ModelTree(BaseModel):
    """Model tree schema."""

    adapters_count: int = 0
    finetunes_count: int = 0
    merges_count: int = 0
    quantizations_count: int = 0


# Schemas related to Model Security Scan Results


class ModelIssue(BaseModel):
    """Model issue schema."""

    title: str
    severity: str
    description: str
    source: str


class ModelSecurityScanResultCreate(BaseModel):
    """Model security scan result create schema."""

    model_id: UUID4
    status: ModelSecurityScanStatusEnum
    total_issues: int
    total_scanned_files: int
    total_skipped_files: int
    scanned_files: list[str]
    low_severity_count: int
    medium_severity_count: int
    high_severity_count: int
    critical_severity_count: int
    model_issues: dict


class ModelSecurityScanResult(ModelSecurityScanResultCreate):
    """Model security scan result schema."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID4
    created_at: datetime
    modified_at: datetime


class ModelDetailSuccessResponse(SuccessResponse):
    """Model detail success response schema."""

    model: ModelDetailResponse
    scan_result: ModelSecurityScanResult | None = None
    eval_result: dict | None = None  # TODO: integrate actual eval result
    model_tree: ModelTree
    endpoints_count: int


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


class CreateLocalModelWorkflowRequest(BaseModel):
    """Local model workflow request schema."""

    workflow_id: UUID4 | None = None
    workflow_total_steps: int | None = None
    step_number: int = Field(..., gt=0)
    trigger_workflow: bool = False
    provider_type: ModelProviderTypeEnum | None = None
    proprietary_credential_id: UUID4 | None = None
    name: str | None = None
    uri: str | None = None
    author: str | None = None
    tags: list[Tag] | None = None
    icon: str | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "CreateLocalModelWorkflowRequest":
        """Validate the fields of the request."""
        if self.workflow_id is None and self.workflow_total_steps is None:
            raise ValueError("workflow_total_steps is required when workflow_id is not provided")

        if self.workflow_id is not None and self.workflow_total_steps is not None:
            raise ValueError("workflow_total_steps and workflow_id cannot be provided together")

        # Validate proprietary_credential_id based on provider_type
        if (
            self.provider_type is not None
            and self.provider_type != ModelProviderTypeEnum.HUGGING_FACE
            and self.proprietary_credential_id is not None
        ):
            raise ValueError("proprietary_credential_id should be None for non-HuggingFace providers")

        if (
            self.provider_type is not None
            and self.provider_type == ModelProviderTypeEnum.HUGGING_FACE
            and self.icon is not None
        ):
            raise ValueError("Icon is not supported for HuggingFace models")

        # Validate provider type
        if self.provider_type and self.provider_type == ModelProviderTypeEnum.CLOUD_MODEL:
            raise ValueError("Cloud model provider type not supported for local model workflow")

        return self


class CreateLocalModelWorkflowSteps(BaseModel):
    """Create cluster workflow step data schema."""

    provider_type: ModelProviderTypeEnum | None = None
    proprietary_credential_id: UUID4 | None = None
    name: str | None = None
    icon: str | None = None
    uri: str | None = None
    author: str | None = None
    tags: list[Tag] | None = None
    provider_id: UUID4 | None


class EditModel(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = Field(None, max_length=300)
    tags: List[Tag] | None = None
    tasks: List[Task] | None = None
    icon: str | None = None
    paper_urls: List[HttpUrl] | None = None
    github_url: HttpUrl | None = None
    huggingface_url: HttpUrl | None = None
    website_url: HttpUrl | None = None
    license_file: UploadFile | None = None
    license_url: HttpUrl | None = None

    @field_validator("name", mode="before")
    def validate_name(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            value = value.strip()
            if len(value) == 0:
                raise ValueError("Model name cannot be empty or only whitespace.")
        return value

    @model_validator(mode="before")
    def validate_license(cls, values):
        license_file = values.get("license_file")
        license_url = values.get("license_url")

        # Ensure only one of license_file or license_url is provided
        if license_file and license_url:
            raise ValueError("Please provide either a license file or a license URL, but not both.")

        if license_file:
            filename = license_file.filename
            allowed_extensions = ["pdf", "txt", "doc", "docx", "md"]

            if not filename or "." not in filename:
                raise ValueError("File does not have a valid extension")

            # Get the file extension from the filename
            file_extension = filename.split(".")[-1].lower()

            # Check if the file extension is in the allowed list
            if file_extension not in allowed_extensions:
                raise ValueError("Invalid file extension for license file")
        return values

    @field_serializer("github_url", "huggingface_url", "website_url", "license_url")
    def str_url(self, url: HttpUrl | None) -> str:
        return str(url) if url else None

    @field_serializer("paper_urls")
    def str_paper_urls(self, urls: List[HttpUrl] | None) -> List[str]:
        return [str(url) for url in urls] if urls else urls


class ModelResponse(BaseModel):
    """Model response schema."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID4
    name: str
    author: str | None = None
    modality: ModalityEnum
    source: str
    uri: str
    created_user: UserInfo | None = None
    model_size: int | None = None
    tasks: list[Task] | None = None
    tags: list[Tag] | None = None
    icon: str | None = None
    description: str | None = None
    provider_type: ModelProviderTypeEnum
    created_at: datetime
    modified_at: datetime
    provider: Provider | None = None
    is_present_in_model: bool | None = None


class ModelListResponse(BaseModel):
    """Model list response schema."""

    model: ModelResponse
    endpoints_count: int | None = None


class ModelPaginatedResponse(PaginatedSuccessResponse):
    """Model paginated response schema."""

    models: list[ModelListResponse] = []


class ModelFilter(BaseModel):
    """Filter model schema for filtering models based on specific criteria."""

    model_config = ConfigDict(protected_namespaces=())

    name: str | None = None
    source: CredentialTypeEnum | None = None
    model_size_min: int | None = Field(None, ge=0, le=500)
    model_size_max: int | None = Field(None, ge=0, le=500)
    provider_type: ModelProviderTypeEnum | None = None
    table_source: Literal["cloud_model", "model"] = "cloud_model"

    @field_validator("source")
    @classmethod
    def change_to_string(cls, v: CredentialTypeEnum | None) -> str | None:
        """Convert the source enum value to a string."""
        return v.value if v else None

    @field_validator("model_size_min", "model_size_max")
    @classmethod
    def convert_to_billions(cls, v: Optional[int]) -> Optional[int]:
        """Convert the input value to billions."""
        if v is not None:
            return v * 1000000000  # Convert to billions
        return v


# Cloud model related schemas


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


class TagsListResponse(PaginatedSuccessResponse):
    """Response schema for tags list."""

    tags: List[Tag] = Field(..., description="List of matching tags")


class TasksListResponse(PaginatedSuccessResponse):
    """Response schema for tasks list."""

    tasks: List[Task] = []


class ModelAuthorResponse(PaginatedSuccessResponse):
    """Response schema for searching tags by name."""

    authors: List[str] = Field(..., description="List of matching authors")


class ModelAuthorFilter(BaseModel):
    """Filter schema for model authors."""

    author: str | None = None


# Schemas related to Paper Published


class PaperPublishedCreate(BaseModel):
    """Paper Published Create Schema."""

    title: str | None = None
    authors: list[str] | None = None
    url: str | None = None
    model_id: UUID4


# Schemas related to Model Licenses


class ModelLicensesCreate(BaseModel):
    """Model Licenses Create Schema."""

    name: str | None = None
    url: str | None = None
    path: str | None = None
    faqs: list[dict] | None = None
    model_id: UUID4


# Local model related schemas


class LocalModelScanRequest(BaseModel):
    """Local model scan request schema."""

    workflow_id: UUID4 | None = None
    workflow_total_steps: int | None = None
    step_number: int = Field(..., gt=0)
    trigger_workflow: bool = False
    model_id: UUID4 | None = None

    @model_validator(mode="after")
    def validate_fields(self) -> "LocalModelScanRequest":
        """Validate the fields of the request."""
        if self.workflow_id is None and self.workflow_total_steps is None:
            raise ValueError("workflow_total_steps is required when workflow_id is not provided")

        if self.workflow_id is not None and self.workflow_total_steps is not None:
            raise ValueError("workflow_total_steps and workflow_id cannot be provided together")

        # Check if at least one of the other fields is provided
        other_fields = [self.model_id]
        required_fields = ["model_id"]
        if not any(other_fields):
            raise ValueError(f"At least one of {', '.join(required_fields)} is required")

        return self


class LocalModelScanWorkflowStepData(BaseModel):
    """Local model scan workflow step data schema."""

    model_id: UUID4 | None
