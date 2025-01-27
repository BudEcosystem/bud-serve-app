from datetime import datetime

from pydantic import UUID4, BaseModel, ConfigDict, Field

from budapp.commons.schemas import PaginatedSuccessResponse, SuccessResponse, Tag
from budapp.model_ops.schemas import CloudModel, Model, ModelSecurityScanResult, Provider

from ..commons.constants import ModelProviderTypeEnum, WorkflowStatusEnum, WorkflowTypeEnum, VisibilityEnum
from ..endpoint_ops.schemas import EndpointResponse


class RetrieveWorkflowStepData(BaseModel):
    """Workflow step data schema."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    provider_type: ModelProviderTypeEnum | None = None
    provider: Provider | None = None
    cloud_model: CloudModel | None = None
    cloud_model_id: UUID4 | None = None
    provider_id: UUID4 | None = None
    model_id: UUID4 | None = None
    model: Model | None = None
    workflow_execution_status: dict | None = None
    leaderboard: list | dict | None = None
    name: str | None = None
    ingress_url: str | None = None
    create_cluster_events: dict | None = None
    delete_cluster_events: dict | None = None
    delete_endpoint_events: dict | None = None
    model_security_scan_events: dict | None = None
    bud_simulator_events: dict | None = None
    icon: str | None = None
    uri: str | None = None
    author: str | None = None
    tags: list[Tag] | None = None
    model_extraction_events: dict | None = None
    description: str | None = None
    security_scan_result_id: UUID4 | None = None
    security_scan_result: ModelSecurityScanResult | None = None
    endpoint: EndpointResponse | None = None
    additional_concurrency: int | None = None
    bud_serve_cluster_events: dict | None = None


class RetrieveWorkflowDataResponse(SuccessResponse):
    """Retrieve Workflow Data Response."""

    workflow_id: UUID4
    status: WorkflowStatusEnum
    current_step: int
    total_steps: int
    reason: str | None = None
    workflow_steps: RetrieveWorkflowStepData | None = None


class WorkflowResponse(SuccessResponse):
    """Workflow response schema."""

    model_config = ConfigDict(
        populate_by_name=True,
    )

    id: UUID4 = Field(alias="workflow_id")
    total_steps: int = Field(..., gt=0)
    status: WorkflowStatusEnum
    current_step: int
    reason: str | None = None


class Workflow(BaseModel):
    """Workflow schema."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID4
    title: str | None = None
    icon: str | None = None
    tag: str | None = None
    progress: dict | None = None
    workflow_type: WorkflowTypeEnum
    total_steps: int = Field(..., gt=0)
    status: WorkflowStatusEnum
    current_step: int
    reason: str | None = None
    created_at: datetime
    modified_at: datetime


class WorkflowListResponse(PaginatedSuccessResponse):
    """Workflow list response schema."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    workflows: list[Workflow]


class WorkflowFilter(BaseModel):
    """Workflow filter schema."""

    workflow_type: WorkflowTypeEnum | None = None


class WorkflowUtilCreate(BaseModel):
    """Workflow create schema."""

    workflow_type: WorkflowTypeEnum
    title: str
    icon: str | None = None
    total_steps: int | None = None
    tag: str | None = None
    visibility: VisibilityEnum = VisibilityEnum.PUBLIC
